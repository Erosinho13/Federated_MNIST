import sys
import copy
import torch
import random
import threading
from tqdm import tqdm
from threading import Thread
from torch.utils.data import DataLoader


class ManageClientThread(Thread):

    def __init__(self, k, k_lens, ws, model):
        super().__init__()
        self.k = k
        self.k_lens = k_lens
        self.ws = ws
        self.model = model

    def run(self):
        self.k_lens.append(len(self.k))
        self.k.updateModel(self.model)
        self.ws.append(self.k.train())


class Server:
    
    def __init__(self, clients, model, num_rounds, C, K,
                 test_dataset=None, batch_size=None, seed=None):
        
        self.clients = clients
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_rounds = num_rounds
        self.m = int(round(max(C*K, 1), 0))
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)

        if seed != None:
            random.seed(seed)

    def __trainClients(self, k_lens, ws, St):

        try:
            threads = [ManageClientThread(k, k_lens, ws, self.model) for k in St]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        except KeyboardInterrupt:
            print(f"\ntid={str(threading.get_ident())[-7:]} quitting...")
            for thread in threads:
                print(f"Ending thread {thread}")
                thread.k.stop_flag = True
                thread.join()
                print("Ended")
            raise KeyboardInterrupt()

    def __averaging(self, k_lens, ws):

        w_avg = copy.deepcopy(ws[0])

        for key in w_avg.keys():
            w_avg[key] *= k_lens[0]
            for i in range(1, len(ws)):
                w_avg[key] += ws[i][key]*k_lens[i]
            w_avg[key] = w_avg[key]/sum(k_lens)

        return w_avg

    def __test(self):

        self.model.eval()

        tot_correct_predictions = 0

        for images, targets in tqdm(self.test_dataloader, position=0, leave=True, file=sys.stdout):

            images = images.to(self.device)
            targets = targets.to(self.device)

            output_probs = self.model(images)

            predictions = torch.argmax(output_probs, dim=1)
            correct_predictions = torch.sum(predictions == targets).item()
            tot_correct_predictions += correct_predictions

        accuracy = tot_correct_predictions/(len(self.test_dataset))*100

        return accuracy

    def federatedAveraging(self, test_during_training=False, save_best=False, path=None):
        
        self.test_accs = []
        best_test_acc = 0

        for t in range(self.num_rounds):
            
            # print(f"ROUND {t+1}/{self.num_rounds}\n\n")
            print(f"\ntid={str(threading.get_ident())[-7:]} - ROUND {t+1}/{self.num_rounds}\n")

            ws = []
            k_lens = []

            St = random.sample(self.clients, self.m)

            self.__trainClients(k_lens, ws, St)
            w_avg = self.__averaging(k_lens, ws)    
            
            self.model.load_state_dict(w_avg)
            
            if test_during_training:
                
                print("\nTESTING...")
                test_accuracy = self.__test()
                print(f"tid={str(threading.get_ident())[-7:]}: Test Accuracy={round(test_accuracy, 2)}%")

                self.test_accs.append(test_accuracy)

                if save_best:

                    if test_accuracy > best_test_acc:

                        print("New best model found!")

                        best_test_acc = test_accuracy
                        torch.save(self.model.state_dict(), path)