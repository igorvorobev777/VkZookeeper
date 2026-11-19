import argparse
import csv
import itertools
import json
import logging
import os
import pickle
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from kazoo.client import KazooClient
from kazoo.exceptions import BadVersionError, NodeExistsError

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "input" / "devops_vacancies.csv"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

CHUNK_SIZE = 200
CHUNK_DELAY_SEC = 5

ZK_HOSTS = os.environ.get("ZK_HOSTS", "127.0.0.1:2181,127.0.0.1:2182,127.0.0.1:2183")
SESSION_TIMEOUT = 10.0

LEADER_NODE = "/homework4/leader"
CHECKPOINT_ZK_NODE = "/homework4/checkpoint_meta"
RESULT_POINTER_NODE = "/homework4/result_pointer"
CHECKPOINT_FILE = CHECKPOINT_DIR / "mapreduce_checkpoint.pkl"


def setup_logger(coord_id: str, run_tag: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{run_tag}_{coord_id}.log"

    logger = logging.getLogger(coord_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_skills(value: str):
    if not value:
        return []
    return sorted({
        skill.strip().lower()
        for skill in value.split(";")
        if skill.strip()
    })


def count_combinations(rows):
    counts = Counter()
    for row in rows:
        skills = parse_skills(row["key_skills"])
        for size in (1, 2, 3):
            for combination in itertools.combinations(skills, size):
                counts[combination] += 1
    return counts


def open_csv_rows(logger: logging.Logger):
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {INPUT_CSV}")

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    logger.info(f"Загружено {len(rows)} строк вакансий из {INPUT_CSV}")
    return rows


def save_checkpoint(zk: KazooClient, processed_rows: int, counts: Counter, coord_id: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp_file, "wb") as f:
        pickle.dump({"processed_rows": processed_rows, "counts": counts}, f)
    os.replace(tmp_file, CHECKPOINT_FILE)

    meta = json.dumps({"processed_rows": processed_rows, "saved_by": coord_id}).encode("utf-8")
    if zk.exists(CHECKPOINT_ZK_NODE):
        zk.set(CHECKPOINT_ZK_NODE, meta)
    else:
        zk.create(CHECKPOINT_ZK_NODE, meta, makepath=True)


def load_checkpoint(logger: logging.Logger):
    if not CHECKPOINT_FILE.exists():
        return 0, Counter()
    with open(CHECKPOINT_FILE, "rb") as f:
        data = pickle.load(f)
    processed_rows = data["processed_rows"]
    counts = data["counts"]
    logger.info(
        f"Найден чекпоинт: уже обработано {processed_rows} строк, "
        f"накоплено {len(counts)} уникальных комбинаций"
    )
    return processed_rows, counts


def save_result_csv(filename: str, counts: Counter) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    with open(out_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["type", "combination", "frequency"])
        for combination, frequency in sorted(
            counts.items(),
            key=lambda item: (len(item[0]), -item[1], item[0]),
        ):
            writer.writerow([len(combination), "; ".join(combination), frequency])
    return out_path


def reset_state(zk: KazooClient, logger: logging.Logger):
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    for node in (LEADER_NODE, CHECKPOINT_ZK_NODE, RESULT_POINTER_NODE):
        if zk.exists(node):
            zk.delete(node)
    zk.create(RESULT_POINTER_NODE, b"init", makepath=True)
    logger.info("Состояние в ZooKeeper и локальный чекпоинт очищены. Указатель результата создан (version=0).")


def acquire_leadership(zk: KazooClient, coord_id: str, logger: logging.Logger) -> int:
    zk.ensure_path("/homework4")
    if not zk.exists(RESULT_POINTER_NODE):
        try:
            zk.create(RESULT_POINTER_NODE, b"init", makepath=True)
        except NodeExistsError:
            pass

    while True:
        try:
            zk.create(LEADER_NODE, coord_id.encode("utf-8"), ephemeral=True)
            _, pointer_stat = zk.get(RESULT_POINTER_NODE)
            expected_version = pointer_stat.version
            logger.info(
                f"СТАЛ ЛИДЕРОМ (создан эфемерный узел {LEADER_NODE}). "
                f"Зафиксирована эпоха (expected_version указателя) = {expected_version}"
            )
            return expected_version
        except NodeExistsError:
            gone_event = threading.Event()

            def on_leader_change(event):
                gone_event.set()

            stat = zk.exists(LEADER_NODE, watch=on_leader_change)
            if stat is None:
                continue
            current_leader, _ = zk.get(LEADER_NODE)
            logger.info(
                f"Лидерство занято ({current_leader.decode('utf-8')}). "
                f"Ожидание освобождения {LEADER_NODE} в режиме запасного координатора..."
            )
            gone_event.wait()
            logger.info(f"Узел {LEADER_NODE} освободился! Пробуем перехватить лидерство...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Идентификатор координатора (coord-1 или coord-2)")
    parser.add_argument("--reset", action="store_true", help="Очистить чекпоинты и узлы перед запуском")
    args = parser.parse_args()

    fencing_enabled = os.environ.get("FENCING", "0") == "1"
    run_tag = "fencing" if fencing_enabled else "no_fencing"
    logger = setup_logger(args.id, run_tag)

    logger.info(
        f"Запуск координатора {args.id} | PID={os.getpid()} | "
        f"FENCING={'1 (включена)' if fencing_enabled else '0 (выключена)'} | "
        f"session_timeout={SESSION_TIMEOUT}s"
    )

    zk = KazooClient(hosts=ZK_HOSTS, timeout=SESSION_TIMEOUT)
    zk.start()

    if args.reset:
        reset_state(zk, logger)

    expected_version = acquire_leadership(zk, args.id, logger)

    rows = open_csv_rows(logger)
    total_rows = len(rows)
    processed_rows, total_counts = load_checkpoint(logger)

    if processed_rows > 0:
        logger.info(f"Продолжаю расчёт с чекпоинта: строка {processed_rows} из {total_rows}")
    else:
        logger.info(f"Начинаю расчёт с нуля: всего {total_rows} строк, шаг чанка {CHUNK_SIZE}")

    while processed_rows < total_rows:
        next_end = min(processed_rows + CHUNK_SIZE, total_rows)
        chunk = rows[processed_rows:next_end]

        chunk_counts = count_combinations(chunk)
        total_counts.update(chunk_counts)
        processed_rows = next_end

        try:
            save_checkpoint(zk, processed_rows, total_counts, args.id)
        except Exception as e:
            logger.warning(f"Не удалось обновить метаданные чекпоинта в ZK ({e}), продолжаем локальный расчёт")

        logger.info(
            f"Обработан батч до строки {processed_rows}/{total_rows} | "
            f"Уникальных комбинаций сейчас: {len(total_counts)}"
        )

        if processed_rows < total_rows:
            time.sleep(CHUNK_DELAY_SEC)

    size_1 = sum(1 for k in total_counts if len(k) == 1)
    size_2 = sum(1 for k in total_counts if len(k) == 2)
    size_3 = sum(1 for k in total_counts if len(k) == 3)
    logger.info(
        f"Расчёт завершён! Контрольные числа (совпадение с ДЗ 3): "
        f"Всего комбинаций = {len(total_counts)} "
        f"(одиночных: {size_1}, пар: {size_2}, троек: {size_3})"
    )

    result_filename = f"mapreduce_counts_{run_tag}_{args.id}.csv"
    out_path = save_result_csv(result_filename, total_counts)

    pointer_payload = json.dumps({
        "author": args.id,
        "file": str(out_path.name),
        "total_combinations": len(total_counts),
        "pairs_count": size_2,
        "fencing": fencing_enabled,
    }).encode("utf-8")

    try:
        if fencing_enabled:
            logger.info(
                f"Публикация результата с проверкой эпохи (FENCING=1, expected_version={expected_version})..."
            )
            new_stat = zk.set(RESULT_POINTER_NODE, pointer_payload, version=expected_version)
        else:
            logger.info("Публикация результата БЕЗ защиты (FENCING=0, version=-1)...")
            new_stat = zk.set(RESULT_POINTER_NODE, pointer_payload, version=-1)

        logger.info(
            f"записал результат -> {RESULT_POINTER_NODE} "
            f"(новая версия указателя version={new_stat.version}, файл={out_path.name})"
        )
    except BadVersionError as e:
        logger.error(
            f"ОТКЛОНЕНО ЗАЩИТОЙ ПО ЭПОХЕ: BadVersionError при записи {RESULT_POINTER_NODE} "
            f"(ожидалась версия {expected_version}): {e}"
        )

    data, final_stat = zk.get(RESULT_POINTER_NODE)
    logger.info(
        f"Текущее состояние указателя {RESULT_POINTER_NODE}: "
        f"version={final_stat.version}, data={data.decode('utf-8')}"
    )

    zk.stop()
    zk.close()


if __name__ == "__main__":
    main()