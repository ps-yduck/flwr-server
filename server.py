# from typing import Any, Callable, Dict, List, Optional, Tuple
# import flwr as fl
# import tensorflow as tf
# import datetime

# from pymongo import MongoClient


# def main() -> None:

#     def get_database():

#         # Provide the mongodb atlas url to connect python to mongodb using pymongo
#         CONNECTION_STRING = "mongodb://localhost:27017"

#         # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
#         try:
#             client = MongoClient(CONNECTION_STRING)
#             print("Connected to MongoDB")
#         except:
#             print("Could not connect to MongoDB")

#         # Create the database for our example (we will use the same database throughout the tutorial
#         return client['flwr_server']

#     db = get_database()
#     user_details = db['user_details']
#     # Create strategy

#     class MyStrategy(fl.server.strategy.FedAvgAndroid):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.client_rounds = {}
#             self.start_time = ""

#         def aggregate_fit(self, server_round, results, failures):
#             # beaware this also prints a very big weight matrix which is send in the results
#             # for client, result in results:

#             #     client_id = client.remote.GetClientId(fl.common.ProtoEmpty()).value
#             #     print("client_id", client_id)
#             # if "hamza" not in self.client_rounds:
#             #     self.client_rounds["hamza"] = []
#             #     self.client_rounds["hamza"].append(server_round)
#             #     user_details.insert_one(
#             #         {"client_id": "hamza", "rounds": self.client_rounds["hamza"]})
#             # else:
#             #     self.client_rounds["hamza"].append(server_round)
#             #     user_details.update_one({"client_id": "hamza"}, {
#             #                             "$set": {"rounds": self.client_rounds["hamza"]}})
#             for result in results:

#                 if result[1].metrics['android_id'] not in self.client_rounds:
#                     self.client_rounds[result[1].metrics['android_id']] = []
#                     self.client_rounds[result[1].metrics['android_id']].append(
#                         {"round": server_round, "start_time": result[1].metrics['start_time'], "end_time": result[1].metrics['end_time']})
#                     user_details.insert_one({"client_id": result[1].metrics['android_id'], "client_device": result[
#                                             1].metrics['device_model'], "rounds": self.client_rounds[result[1].metrics['android_id']]})
#                 else:
#                     self.client_rounds[result[1].metrics['android_id']].append(
#                         {"round": server_round, "start_time": result[1].metrics['start_time'], "end_time": result[1].metrics['end_time']})
#                     user_details.update_one({"client_id": result[1].metrics['android_id']}, {
#                         "$set": {"rounds": self.client_rounds[result[1].metrics['android_id']]}})

#                 # print("result", result[1].metrics)
#                 # print("start_time", self.start_time)
#             # print("results", results)
#             # print("failures", failures)
#             return super().aggregate_fit(server_round, results, failures)
#             # Aggregate results

#     strategy = MyStrategy(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2,
#         evaluate_fn=None,
#         on_fit_config_fn=fit_config,
#         initial_parameters=None,
#     )

#     # Start Flower server for 10 rounds of federated learning
#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=4),
#         strategy=strategy,
#     )


# def fit_config(server_round: int):
#     """Return training configuration dict for each round.

#     Keep batch size fixed at 32, perform two rounds of training with one
#     local epoch, increase to two local epochs afterwards.
#     """
#     config = {
#         "batch_size": 32,
#         "local_epochs": 5,
#     }
#     return config


# if __name__ == "__main__":
#     main()


from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
import os



def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server(
        server_address=os.environ.get('PORT', 3001),
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 5,
    }
    return config


if __name__ == "__main__":
    main()
