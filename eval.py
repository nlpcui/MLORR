import sys, pickle, json, argparse, os, torch, math, copy
import numpy as np
from train import NNRegressor
from sklearn.metrics import mean_squared_error
from read_data import DataProcessor, ABO3Dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset


def trunc(value, min_=np.float64(-2.05), max_=np.float64(0.5)):
    if value <= min_:
        return min_
    elif value >= max_:
        return max_
    else:
        return value


def eval_ftr_importance(model, eval_dataset, test_dims, run_num=10):
    results = []
    results_p = []
    for i in range(run_num):
        eval_loader = DataLoader(ABO3Dataset(eval_dataset), batch_size=1)
        mse = {dim: 0 for dim in test_dims}
        mse_p = {dim: 0 for dim in test_dims}
        base_result = eval_mse(model, eval_dataset)
        for test_dim in test_dims:
            eval_dataset_ = DataProcessor.mask_dims(
                copy.deepcopy(eval_dataset), test_dim, mask='random')
            eval_loader_ = DataLoader(ABO3Dataset(eval_dataset_), batch_size=1)
            eval_result = eval_mse(model, eval_dataset_)
            mse[test_dim] = eval_result['mse'] - base_result['mse']
            mse_p[test_dim] = eval_result['mse_p'] - base_result['mse_p']

        results.append(mse)
        results_p.append(mse_p)

    rank_items = sorted(average(results).items(),
                        key=lambda x: x[1][0], reverse=True)
    rank_items_p = sorted(average(results_p).items(),
                          key=lambda x: x[1][0], reverse=True)

    return rank_items, rank_items_p


def remove_max_min(array, remove):
    arr_new = copy.deepcopy(array)
    if remove == 'max':
        arr_new.remove(np.max(arr_new))
    else:
        arr_new.remove(np.min(arr_new))
    return arr_new


def average(mse_dicts):
    record = {}
    for mse_dict in mse_dicts:
        for dim in mse_dict:
            if dim not in record:
                record[dim] = []
            record[dim].append(mse_dict[dim])
    for dim in record:
        x = remove_max_min(record[dim], 'max')
        x = remove_max_min(x, 'min')
        record[dim] = (np.mean(x), np.std(x))

    return record

def eval_ml_model(model, test_dataset, new_dataset):
    results = {
        'test_preds': [],
        'new_preds': [],
        'test_mse': 0,
        'test_mse_p': 0,
    }

    test_predictions = model.predict(test_dataset['data_x'])
    test_predictions_p = np.power(10, test_predictions)
    new_predictions = model.predict(new_dataset['data_x'])
    new_predictions_p = np.power(10, new_predictions)

    results['test_mse'] = mean_squared_error(
        test_dataset['data_y'], model.predict(test_dataset['data_x']))
    results['test_mse_p'] = mean_squared_error(
        np.power(10, test_dataset['data_y']), test_predictions_p)
    results['test_preds'] = list(zip(test_dataset['names'], test_dataset['data_y'], np.power(10, test_dataset['data_y']), test_predictions, test_predictions_p))
    results['new_preds'] = list(zip(new_dataset['names'], new_predictions, new_predictions_p))
    
    return results


def eval_mse(model, dataset, use_trunc=False):
    results = {
        'preds': [],
        'mse': 0,
        'mse_p': 0,
    }

    data_loader = DataLoader(ABO3Dataset(dataset), batch_size=1)

    preds = []
    preds_p = []
    true = []
    true_p = []
    names = []
    for idx, (data_x, data_y, name) in enumerate(data_loader):
        if trunc:
            y_ = trunc(model(data_x.float()).detach().numpy()[0][0])
        else:
            y_ = model(data_x.float()).detach().numpy()[0][0]
        y = data_y.detach().numpy()[0][0]
        names.append(name[0])
        preds.append(float(y_))
        preds_p.append(float(math.pow(10, y_)))
        true.append(float(y))
        true_p.append(float(math.pow(10, y)))

    results['preds'] = list(zip(names, preds, preds_p, true, true_p))
    results['mse'] = mean_squared_error(true, preds)
    results['mse_p'] = mean_squared_error(true_p, preds_p)

    return results


def eval_ann(model, full_dataset, test_dataset, new_dataset):
    test_results = eval_mse(model, test_dataset)
    new_results = eval_mse(model, new_dataset, use_trunc=True)

    test_dims_comb = [(7, 11), (8, 12), (9, 13), (10, 14)]
    test_dims_single = [(i,) for i in range(7, 16)]

    _, combine_importance_p = eval_ftr_importance(model, full_dataset, test_dims=test_dims_comb)
    _, single_importance_p = eval_ftr_importance(model, full_dataset, test_dims=test_dims_single)

    results = {
        'single_importance': single_importance_p,
        'combine_importance': combine_importance_p,
        'test_preds': test_results['preds'],
        'test_mse': test_results['mse'],
        'test_mse_p': test_results['mse_p'],
        'new_oxide_preds': new_results['preds']
    }

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ann')  # [ols, lasso, ridge, svr, rf, gpr, ann1, ann2, ann3]
    parser.add_argument('--output_dir', default='data/results')
    parser.add_argument('--data', type=int, default=700) # [700, 650]]
    parser.add_argument('--model_params', type=str, default='data/model_params.json')

    args = parser.parse_args()

    data_file = 'data/dataset.xlsx'
    reverse_y = 'log'
    normalize_mask_dims = [i for i in range(7)]
    feature_mask_dims = [-1]  # drop u

    processor = DataProcessor(data_file=data_file,
                            normalize_mask_dims=normalize_mask_dims)

    test_dataset = processor.get_dataset(
        args.data, split='test', mask_dims=feature_mask_dims)
    DataProcessor.change_y(test_dataset, reverse_y)

    full_dataset = processor.get_dataset(
        args.data, split='full', mask_dims=feature_mask_dims)
    DataProcessor.change_y(full_dataset, reverse_y)

    new_dataset = processor.get_dataset(
        args.data, split='new', mask_dims=feature_mask_dims)


    nn_activations = {
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh()
    }

    with open(args.model_params, 'r') as fp:
        best_parameters = json.loads(''.join(fp.readlines()))

    with open(args.model_params, 'r') as fp:
        model_params = json.loads("".join(fp.readlines()))
    
    model_file = '{}.md'.format(os.path.join(args.output_dir, args.model))

    if args.model in ['ols', 'lasso', 'ridge', 'svr', 'rf', 'gpr', 'esnet']:
        with open(model_file, 'rb') as fp:
            model = pickle.load(fp)
        results = eval_ml_model(model, test_dataset, new_dataset)

    elif args.model in ['ann_1', 'ann_2', 'ann_3']:
        model = NNRegressor(dims=best_parameters[args.model]['dims'] , activation=nn_activations[best_parameters[args.model]['activation']])
        model.load_state_dict(torch.load(model_file))
        results = eval_ann(model, full_dataset, test_dataset, new_dataset)

    with open("{}_test_result.json".format(os.path.join(args.output_dir, args.model)), 'w') as fp:
        fp.write(json.dumps(results))
    
