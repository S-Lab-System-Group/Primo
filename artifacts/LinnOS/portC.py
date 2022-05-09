import joblib
import argparse
from sklearn_porter import Porter


def main(args):
    drive = args.drive
    cfile = f"primodrive{drive}"

    model = joblib.load(f"./traces/primodt_{drive}.pkl")

    porter = Porter(model, language="c")
    output = porter.export()  # embed_data=True

    with open(f"./traces/{cfile}.c", "w") as f:
        f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primo-LinnOS")
    parser.add_argument("-d", "--drive", default=0, type=int, help="Drive ID")

    args = parser.parse_args()
    main(args)
