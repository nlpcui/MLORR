# MLORR
This repo contains the source code of "*A combined ionic Lewis acid descriptor and machine-learning approach to prediction of efficient oxygen reduction electrodes for ceramic fuel cells*" (Accpeted by **Nature Energy**). In this study, we constructed a small but high-quality ABO3 perovskite dataset (`data/dataset.xlsx`), based on which we built up various regression models to explore the potential patterns hidden behind the data.

## Requirements
The code is based on Python 3.8 (other Python 3+ versions may work as well). Before running the code, make sure all denepdencies are propoerly installed via `pip3 install -r requirements.txt`.

## Reproduction
**To train the regression models** used in our paper, run `python train.py --model <model_name> --data <dataset> --output_dir <output_dir> --model_params <param_file>`.

**arguments:**
+ `<model_name>`: options: `[ols, lasso, ridge, svr, rf, gpr, ann_1, ann_2, ann_3]`.
+ `<dataset>`: options: `[650, 700]`. Our dataset has two subsets with temperature of 650 and 700. Our main conclusions are based on dataset of temperature 700.
+ `<output_dir>`: directory where the training results (`<model_name>_train_result.json`) and model files (`<model_name>.md`) are saved. default "`data/results/`".
+ `<param_file>`: the hyper-parameter setups of models. default "`model_params.json`"

For example, `python train.py --model svr --data 700 --output_dir data/results --model_params model_params.json` will train a SVR model on the 700 subset using hyperparameters sepcified in model_params.json, with the training results and SVR model saved in "data/results/svr_train_result.json" and "data/results/svr.md", respectively.  

**To evaluate models** on the test set, run `python eval.py` with the same arguments.

For more details and the full data, please refer to our paper.