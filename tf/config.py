REDIS_HOST = 'local'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_USERNAME = ""
REDIS_PASSWORD = ""
DEVICE = 'cpu'

# server_config = Config()
# device = config.DEVICE
# if 'gpu' in device.lower():
# commands = ['docker', 'run', '-p', '6379:6379', '--gpus', 'all', '--rm', 'redisai/redisai:edge-gpu-bionic']
# else:
# commands = ['docker', 'run', '-p', '6379:6379', '--rm', 'redislabs/redisai:edge-cpu-bionic']
# proc = subprocess.Popen(commands)    
# self.con = Client(**server_config)
# start_time = time.time()
# prev_num_interval = 0
# while True:
# logger.info("Launching RedisAI docker container")
# try:
#     if self.con.ping():
#         break
# except redis.exceptions.ConnectionError:
#     num_interval, _ = divmod(time.time() - start_time, 10)
#     if num_interval > prev_num_interval:
#         prev_num_interval = num_interval
#         try:
#             proc.communicate(timeout=0.1)
#         except subprocess.TimeoutExpired:
#             pass
#         else:
#             raise RuntimeError("Could not start the RedisAI docker container. You can "
#                             "try setting up RedisAI locally by (by following the "
#                             "documentation https://oss.redislabs.com/redisai/quickstart/)"
#                             " and call the ``create`` API with target_uri as given in "
#                             "the example command below (this will set the host as "
#                             "localhost and port as 6379)\n\n"
#                             "    mlflow deployments create -t redisai -m <modeluri> ...\n\n")
#     time.sleep(0.2)