import tensorflow as tf
import subprocess
import logging
import os

MAX_STEPS = 300000
STEPS_PER_VALIDATION = 1000
STEPS_PER_CHECKPOINT = 20000
TEST_THRESHOLD = 200000

train_params = {
    "--steps_per_validation": STEPS_PER_VALIDATION,
    "--steps_per_checkpoint": STEPS_PER_CHECKPOINT,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    try:
        global_step = tf.contrib.framework.load_variable("model", "global_step")
    except:
        global_step = 0

    logging.info("Training starts with global_step={}. ".format(global_step))

    while global_step < MAX_STEPS:
        terminate_step = max(global_step + STEPS_PER_CHECKPOINT, TEST_THRESHOLD)

        logging.info("Train from {} to {}. ".format(
            global_step, terminate_step))

        proc = ["python3", "src/summarization.py",
            "--max_iter", str(terminate_step)]
        for key, val in train_params.items():
            proc.append(key)
            proc.append(str(val))
        subprocess.call(proc)

        global_step = terminate_step

        subprocess.call(["python3", "script/test.py"])
