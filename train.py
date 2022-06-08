from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel, RationalQuadratic, Matern, ExpSineSquared
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy, random, sklearn, math, sys, torch, json, argparse, configparser, pickle, os
from read_data import DataProcessor, ABO3Dataset


def trunc(value, min_=np.float64(-2.05), max_=np.float64(0.5)):
    if value <= min_:
        return min_
    elif value >= max_:
        return max_
    else:
        return value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# NN regressor
class NNRegressor(torch.nn.Module):
    def __init__(self, dims, activation=torch.nn.ReLU(), last_act=None):
        super(NNRegressor, self).__init__()
        self.network = torch.nn.ModuleList([])
        for i in range(len(dims)-1):
            self.network.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                self.network.append(activation)
        if not last_act is None:
            self.network.append(last_act)

    def forward(self, x):
        for f in self.network:
            x = f(x)
        return x


def train_ml_model(train_dataset_all, test_dataset, new_dataset=None, model=None):
    mse_val = []
    mse_val_p = []
    for idx in range(train_dataset_all['size']):
        train_dataset, validation_dataset = DataProcessor.loo_validation(
            train_dataset_all, idx)
        model.fit(train_dataset['data_x'], train_dataset['data_y'])
        mse_val.append(mean_squared_error(
            validation_dataset['data_y'], model.predict(validation_dataset['data_x'])))
        mse_val_p.append(mean_squared_error(
            np.power(10, validation_dataset['data_y']), np.power(
                10, model.predict(validation_dataset['data_x']))
        ))
    mse_val_std = np.std(mse_val)
    mse_val_std_p = np.std(mse_val_p)
    mse_val = np.mean(mse_val)
    mse_val_p = np.mean(mse_val_p)

    model.fit(train_dataset_all['data_x'], train_dataset_all['data_y'])

    train_predictions = model.predict(train_dataset_all['data_x'])
    train_predictions_p = np.power(10, train_predictions)
    mse_train = mean_squared_error(
        train_dataset_all['data_y'], train_predictions)
    mse_train_p = mean_squared_error(
        np.power(10, train_dataset_all['data_y']), train_predictions_p)

    results = {
        'mse_train': mse_train,
        'mse_train_p': mse_train_p,
        'mse_val': mse_val,
        'mse_val_p': mse_val_p,
        'mse_val_std': mse_val_std,
        'mse_val_std_p': mse_val_std_p,
        'train_preds': list(zip(train_dataset_all['names'], train_dataset_all['data_y'], np.power(10, train_dataset_all['data_y']), train_predictions, train_predictions_p))
    }

    return results, model


def grid_search_linear(train_dataset_all, test_dataset):
    records = {}
    for l1_weight in [0, 0.01, 0.2, 0.5, 1.0, 2]:
        for l2_weight in [0, 0.2, 0.5, 1.0, 2]:
            model_name = None
            setting = 'l1:{}-l2:{}'.format(l1_weight, l2_weight)
            if l1_weight == 0 and l2_weight == 0:
                linear_model = LinearRegression()
                model_name = 'ols'
            elif l1_weight == 0 and l2_weight > 0:
                linear_model = Ridge(alpha=l2_weight)
                model_name = 'ridge-l1:{}-l2:{}'.format(l1_weight, l2_weight)
            elif l1_weight > 0 and l2_weight == 0:
                linear_model = Lasso(alpha=l1_weight)
                model_name = 'lasso-l1:{}-l2:{}'.format(l1_weight, l2_weight)
            else:
                alpha = l1_weight+2*l2_weight
                l1_ratio = l1_weight/alpha
                linear_model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio)
                model_name = 'es_net-l1:{}-l2:{}-alpha:{}-l1ratio:{}'.format(
                    l1_weight, l2_weight, alpha, l1_ratio)

            records[model_name] = train_ml_model(
                train_dataset_all, test_dataset, model=linear_model)

    output_performance(records)


def grid_search_gpr(train_dataset_all, test_dataset):
    records = {}
    kernels = []
    kernel_names = []
    for constant_value in [0, 0.5, 1, 10]:
        for noise_level in [0, 0.05, 0.5, 1.0]:
            # Matern Kernal (=RBF when nu=0.5)
            for nu in [0.5, 1, 5]:
                for length_scale in [0.5, 1,  5]:
                    kernels.append(WhiteKernel(
                        noise_level)+ConstantKernel(constant_value)+Matern(length_scale=length_scale, nu=nu))
                    kernel_names.append('WhiteKernel({})+ConstantKernel({})+MaternKernel({}, {})'.format(
                        noise_level, constant_value, length_scale, nu))
            # DocProduct Kernel
            for sigma in [0.5, 1.0, 5]:
                kernels.append(WhiteKernel(
                    noise_level)+ConstantKernel(constant_value)+DotProduct(sigma))
                kernel_names.append('WhiteKernel({})+ConstantKernel({})+DotProduct({})'.format(
                    noise_level, constant_value, sigma))
            # RationalQuadratic Kernel
            for length_scale in [0.5, 1.0, 2, 5]:
                kernels.append(WhiteKernel(
                    noise_level)+ConstantKernel(constant_value)+RationalQuadratic(length_scale=length_scale))
                kernel_names.append('WhiteKernel({})+ConstantKernel({})+RationalQuadratic({})'.format(
                    noise_level, constant_value, length_scale))
            # ExpSineSquared Kernel
            for length_scale in [0.5, 1., 2., 5]:
                for periodicity in [0.5, 1, 2, 5]:
                    kernels.append(WhiteKernel(noise_level)+ConstantKernel(constant_value) +
                                   ExpSineSquared(length_scale=length_scale, periodicity=periodicity))
                    kernel_names.append('WhiteKernel({})+ConstantKernel({})+ExpSineSquared({}, {})'.format(
                        noise_level, constant_value, length_scale, periodicity))

    for i in range(len(kernels)):
        model = GaussianProcessRegressor(kernel=kernels[i], random_state=0)
        try:
            records[kernel_names[i]] = train_ml_model(
                train_dataset_all, test_dataset, model)
        except Exception:
            continue
    output_performance(records)


def output_performance(records):
    best_model_val = sorted(records.items(), key=lambda x: x[1]['mse_val_p'])
    best_model_test = sorted(records.items(), key=lambda x: x[1]['mse_test_p'])
    best_model_train = sorted(
        records.items(), key=lambda x: x[1]['mse_train_p'])

    for item in best_model_val:
        print('model_name: {}, mse_train: {}, mse_train_p: {}, mse_val: {}, mse_val_p: {}, mse_test: {}, mse_test_p:{}, mse_val_std: {}, mse_val_std_p: {}'.format(
            item[0],
            item[1]['mse_train'],
            item[1]['mse_train_p'],
            item[1]['mse_val'],
            item[1]['mse_val_p'],
            item[1]['mse_test'],
            item[1]['mse_test_p'],
            item[1]['mse_val_std'],
            item[1]['mse_val_std_p']
        )),


def grid_search_rf(train_dataset_all, test_dataset):
    records = {}
    for n_estimators in [10, 50, 100]:
        for max_depth in [5, 10, 15, None]:
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=3)
            model_name = 'n:{}-depth:{}'.format(n_estimators, max_depth)
            records[model_name] = train_ml_model(
                train_dataset_all, test_dataset, model=rf_model)

    output_performance(records)


def grid_search_svr(train_dataset_all, test_dataset):
    records = {}
    for kernel in ['poly', 'rbf', 'sigmoid']:
        for c in [0.1, 1, 5]:
            for gamma in [0.1, 1, 0.01]:
                if kernel == 'rbf':
                    model = SVR(kernel=kernel, C=c, gamma=gamma)
                else:
                    model = SVR(kernel=kernel)
                model_name = '{}-{}-{}'.format(kernel, c, gamma)
                records[model_name] = train_ml_model(
                    train_dataset_all, test_dataset, model=model)

    output_performance(records)



def train_ann(dims, lr, batch_size, activation, epoch, train_dataset_all, seed):

    results = {
        'train_mse': 0,
        'train_mse_p': 0,
        'train_preds': [],
        'val_mse': 0,
        'val_mse_p': 0,
        'val_mse_std': 0,
        'val_mse_std_p': 0,
    }

    val_mse = []
    val_mse_p = []
    for idx in range(train_dataset_all['size']):
        train_dataset, validation_dataset = DataProcessor.loo_validation(
            train_dataset_all, idx)

        train_loader = DataLoader(ABO3Dataset(train_dataset), batch_size=batch_size if type(
            batch_size) == int else len(train_dataset))
        val_loader = DataLoader(ABO3Dataset(validation_dataset), batch_size=1)

        model = NNRegressor(dims=dims, activation=activation)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch_ in range(epoch):
            for batch_id, (data_x, data_y, name) in enumerate(train_loader):
                optimizer.zero_grad()
                y_ = model(data_x.float())
                loss = criterion(y_, data_y.float())
                loss.backward()
                optimizer.step()

        val_predictions = []
        val_predictions_p = []

        for batch_id, (data_x, data_y, name) in enumerate(val_loader):
            y_pred = trunc(model(data_x.float()).detach().numpy()[0][0])
            y_pred_p = np.power(10, y_pred)
            y_true = data_y.detach().numpy()[0][0],
            y_true_p = np.power(10, y_true)
            val_predictions.append([
                y_pred,
                y_true,
            ])
            val_predictions_p.append([
                y_pred_p,
                y_true_p
            ])
        val_mse.append(np.mean(np.square(np.abs([item[0]-item[1]
                       for item in val_predictions]))))
        val_mse_p.append(
            np.mean(np.square(np.abs([item[0]-item[1] for item in val_predictions_p]))))

    results['val_mse'] = np.mean(val_mse)
    results['val_mse_p'] = np.mean(val_mse_p)
    results['val_mse_std'] = np.std(val_mse)
    results['val_mse_std_p'] = np.std(val_mse_p)

    setup_seed(seed)

    train_all_loader = DataLoader(ABO3Dataset(
        train_dataset_all), batch_size=batch_size if type(batch_size) == int else train_dataset_all['size'])
    test_loader = DataLoader(ABO3Dataset(test_dataset), batch_size=1)
    new_oxide_loader = DataLoader(ABO3Dataset(new_dataset), batch_size=1)

    model = NNRegressor(dims=dims, activation=activation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch_ in range(epoch):
        for batch_id, (data_x, data_y, name) in enumerate(train_all_loader):
            optimizer.zero_grad()
            y_ = model(data_x.float())
            loss = criterion(y_, data_y.float())
            loss.backward()
            optimizer.step()

    train_all_loader = DataLoader(ABO3Dataset(train_dataset_all), batch_size=1)
    train_preds = []
    train_preds_p = []
    train_y = []
    train_y_p = []
    train_names = []
    for idx, (data_x, data_y, name) in enumerate(train_all_loader):
        y_ = model(data_x.float()).detach().numpy()[
            0][0]
        y = data_y.detach().numpy()[0][0]
        train_names.append(name[0])
        train_preds.append(float(y_))
        train_preds_p.append(float(math.pow(10, y_)))
        train_y.append(float(y))
        train_y_p.append(float(math.pow(10, y)))

    results['train_preds'] = list(
        zip(train_names, train_preds, train_preds_p, train_y, train_y_p))

    results['train_mse'] = mean_squared_error(train_y, train_preds)
    results['train_mse_p'] = mean_squared_error(train_y_p, train_preds_p)

    return results, model


def eval_nn(eval_dataloader, model, use_trunc=False):
    pred = []
    pred_p = []
    real = []
    real_p = []
    names = []
    for batch_id, (data_x, data_y, name) in enumerate(eval_dataloader):
        names.append(name[0])
        y_pred = model(data_x.float()).detach().numpy()[0][0]
        if use_trunc:
            y_pred = trunc(y_pred)
        pred.append(y_pred)
        pred_p.append(np.power(10, pred[-1]))
        real.append(data_y.detach().numpy()[0][0])
        real_p.append(np.power(10, real[-1]))
    preds = list(zip(names, real, real_p, pred, pred_p))
    eval_mse = mean_squared_error(real, pred)
    eval_mse_p = mean_squared_error(real_p, pred_p)
    return eval_mse, eval_mse_p, preds


def average(train_dataset, test_dataset):
    ave_train = np.mean(train_dataset['data_y'], axis=-1)
    ave_test = np.mean(test_dataset['data_y'])
    print(train_dataset['data_y'])
    print(ave_train)
    return np.mean(np.abs(train_dataset['data_y'] - ave_train)), np.mean(np.square(np.abs(test_dataset['data_y']-ave_train)))


def grid_search_gbdt(train_dataset_all, test_dataset):
    records = {}
    for n_estimators in [10, 50, 100]:
        for max_depth in [3, 5, 10, 50]:
            for learning_rate in [1e-2, 1e-3, 1e-1]:
                gbdt_model = GradientBoostingRegressor(
                    loss='squared_error', n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                model_name = 'n:{}-depth:{}-lr:{}'.format(
                    n_estimators, max_depth, learning_rate)
                records[model_name] = train_ml_model(
                    train_dataset_all, test_dataset, gbdt_model)

    output_performance(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ann')  # [ols, lasso, ridge, svr, rf, gpr, ann_1, ann_2, ann_3]
    parser.add_argument('--output_dir', type=str, default='data/results')
    parser.add_argument('--data', type=int, default=700) # [700, 650]]
    parser.add_argument('--model_params', type=str, default='data/model_params.json')

    args = parser.parse_args()

    with open(args.model_params, 'r') as fp:
        best_parameters = json.loads(''.join(fp.readlines()))

    data_file = 'data/dataset.xlsx'
    reverse_y = 'log'
    normalize_mask_dims = [i for i in range(7)]
    feature_mask_dims = [-1]  # drop u

    processor = DataProcessor(data_file=data_file,
                            normalize_mask_dims=normalize_mask_dims)
    train_dataset_all = processor.get_dataset(
        args.data, split='train', mask_dims=feature_mask_dims)
    DataProcessor.change_y(train_dataset_all, reverse_y)
    test_dataset = processor.get_dataset(
        args.data, split='test', mask_dims=feature_mask_dims)
    DataProcessor.change_y(test_dataset, reverse_y)
    full_dataset = processor.get_dataset(
        args.data, split='full', mask_dims=feature_mask_dims)
    DataProcessor.change_y(full_dataset, reverse_y)
    new_dataset = processor.get_dataset(
        args.data, split='new', mask_dims=feature_mask_dims)

    gpr_kernels = {
        'DotProduct': DotProduct,
        'RationalQuadratic': RationalQuadratic,
        'ExpSineSquared': ExpSineSquared,
        'Matern': Matern,
    }
    nn_activations = {
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh()
    }

    # load models
    model_map = {
        'ols': LinearRegression(),
        'lasso': Lasso(alpha=best_parameters['lasso']['alpha']),
        'ridge': Ridge(alpha=best_parameters['ridge']['alpha']),
        'svr': SVR(**best_parameters['svr']),
        'rf': RandomForestRegressor(
            n_estimators=best_parameters['rf']['n_estimators'],
            max_depth=best_parameters['rf']['max_depth'],
            random_state=best_parameters['rf']['random_state']
        ),
        'esnet': ElasticNet(**best_parameters['esnet']),
        'gpr': GaussianProcessRegressor(WhiteKernel(best_parameters['gpr']['WhiteKernel'])+ConstantKernel(best_parameters['gpr']['ConstantKernel'])+ gpr_kernels[best_parameters['gpr']['Kernel']](**best_parameters['gpr']['kernel_params'])),
    }


    if args.model in ['ols', 'lasso', 'ridge', 'rf', 'svr', 'gpr', 'esnet']:
        train_dataset_all = DataProcessor.shuffle(train_dataset_all)
        train_result, model = train_ml_model(
            train_dataset_all, 
            test_dataset, 
            model=model_map[args.model]
        )
        with open(os.path.join(args.output_dir, '{}.md'.format(args.model)), 'wb') as fp:
            pickle.dump(model, fp)
    else:
        setup_seed(best_parameters[args.model]['random_state'])
        train_dataset_all = DataProcessor.shuffle(train_dataset_all)
        train_result, model = train_ann(
            dims=best_parameters[args.model]['dims'], 
            lr=best_parameters[args.model]['learning_rate'], 
            batch_size=best_parameters[args.model]['batch_size'], 
            activation=nn_activations[best_parameters[args.model]['activation']], 
            epoch=best_parameters[args.model]['epoch'], 
            train_dataset_all=train_dataset_all, 
            seed=best_parameters[args.model]['random_state']
        )
        torch.save(model.state_dict(), os.path.join(args.output_dir, '{}.md'.format(args.model)))

    with open(os.path.join(args.output_dir, '{}_train_result.json'.format(args.model)), 'w') as fp:
        fp.write(json.dumps(train_result))
