from models.DistMult import DistMult
from models.Complex import Complex
from models.ConvE import ConvE, ConvE_args

from utils.loaders import load_data, get_onehots
from utils.evaluation_metrics import SRR, auprc_auroc_ap
from utils.path_manage import get_files

import torch
from sklearn.utils import shuffle
import argparse
from sklearn.model_selection import train_test_split

def main(model, optimiser, train_data, val_data, epochs, batches):
    mean_reciporical_rank = 0
    break_condition = False
    for epoch in range(epochs):
        # training stage
        model.train()
        objects, subjects, relationships = load_data(train_data, batches)

        for index in range(batches):

            obj = torch.LongTensor(objects[index])
            rel = torch.LongTensor(relationships[index])
            subj = torch.squeeze(torch.LongTensor(subjects[index]))

            optimiser.zero_grad()
            pred = model.forward(e1=obj, rel=rel)
            loss = model.loss(pred, subj)
            loss.backward()
            optimiser.step()

        # evaluation stage

        if epoch % 10 == 0:

            model.eval()
            objects, subjects, relationships = load_data(val_data, batch_number=1)
            total_sum_reciporical_rank = torch.zeros(1)

            obj = torch.squeeze(torch.LongTensor(objects)).unsqueeze(1)
            rel = torch.squeeze(torch.LongTensor(relationships)).unsqueeze(1)
            targets = torch.squeeze(torch.LongTensor(subjects)).unsqueeze(1)

            predictions = model.forward(e1 = obj, rel = rel)
            srr = SRR(predictions, targets)
            total_sum_reciporical_rank = total_sum_reciporical_rank + srr
            print("mean reciporical rank is...")


            print(total_sum_reciporical_rank / len(val_data))
            MRR = total_sum_reciporical_rank / len(val_data)
            if MRR < (mean_reciporical_rank * 1.05):
                torch.save(
                    model,
                    "Model_{model}_Epoch_{epoch}_MRR_{MRR}.pickle".format(
                        model=args.model, epoch=epoch, MRR=MRR
                    ),
                )
                break_condition = True
            else:
                mean_reciporical_rank = MRR

        if epoch % 399 == 0:
            torch.save(
                model,
                "Model_{model}_Epoch_Final_MRR_{MRR}.pickle".format(
                    model=args.model, MRR=MRR
                ),
            )

        if break_condition == True:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="this sets parameters for experiments")
    parser.add_argument(
        "--model", default="DistMult", choices=["DistMult", "Complex", "ConvE"]
    )
    parser.add_argument(
        "--embdim",
        default=100,
        choices=[100, 256, 625, 1024, 1600, 2025],
        type=int,
        help="this is the dimensionality of the embeddings",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batches", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)

    args = parser.parse_args()
    print(args)

    ConvE_args = ConvE_args()

    data, lookup, ASD_dictionary, BCE_dictionary, _, __ = get_files()
    entities = int(len(lookup) / 2)

    # data = shuffle(data)
    test_data = data[:]

    X_train, X_test = train_test_split(test_data, test_size=0.1, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.1111, random_state=1)


    if args.model == "DistMult":
        model = DistMult(
            args=ConvE_args,
            embedding_dim=args.embdim,
            num_entities=entities,
            num_relations=4,
        )
    elif args.model == "Complex":
        model = Complex(
            args=ConvE_args,
            embedding_dim=args.embdim,
            num_entities=entities,
            num_relations=4,
        )
    else:
        model = ConvE(
            args=ConvE_args,
            embedding_dim=args.embdim,
            num_entities=entities,
            num_relations=4,
        )

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    main(
        model=model,
        optimiser=optimiser,
        train_data=X_train,
        val_data=X_val,
        epochs=args.epochs,
        batches=args.batches,
    )
