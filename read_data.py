
import torch, sys, math, scipy, random, json, xlrd, pandas, copy
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, TensorDataset


class ABO3Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data_x = np.array(dataset['data_x'])
        self.data_y = np.array(dataset['data_y']).reshape(-1, 1)
        self.name_list = dataset['names']

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx], self.name_list[idx]

    def __len__(self):
        return len(self.data_x)

class DataProcessor:
    def __init__(self, data_file,  A_site_sheet='A_descriptors', B_site_sheet='B_descriptors', labeled_data_sheet='ABO3', new_oxide_sheet='new_oxides', normalize_mask_dims=[]) -> None:
        self.data_file = data_file
        self.labeled_data_sheet = labeled_data_sheet
        self.A_site_sheet = A_site_sheet
        self.B_site_sheet = B_site_sheet
        self.new_oxide_sheet = new_oxide_sheet

        self.normalize_mask_dims = normalize_mask_dims

        self.elements_A = {}
        self.elements_B = {}
        self.data_raw = []
        self.data_featurized = {} 

        self.data_new = {'data_x': [], 'data_y': [], 'names': []}

        self.read_labeled_data()
        self.read_descriptors()
        self.build_feature()
        self.normalize()
        self.read_new_oxides()

    def read_labeled_data(self, ):
        ABO3 = pandas.read_excel(
            self.data_file, sheet_name=self.labeled_data_sheet)
        for row_index in range(ABO3.shape[0]):
            material = str(ABO3.iloc[row_index, 0])
            A_site_eles, B_site_eles = self.__parse_str(material)
            data = [
                A_site_eles,
                B_site_eles,
                {},
                {
                    700: ABO3.iloc[row_index, 8],
                    650: ABO3.iloc[row_index, 9],
                    600: ABO3.iloc[row_index, 10],
                    550: ABO3.iloc[row_index, 11],
                    500: ABO3.iloc[row_index, 12],
                }
            ]

            data[2][700] = ABO3.iloc[row_index, 2]
            data[2][650] = ABO3.iloc[row_index, 3]
            data[2][600] = ABO3.iloc[row_index, 4]
            data[2][550] = ABO3.iloc[row_index, 5]
            data[2][500] = ABO3.iloc[row_index, 6]

            self.data_raw.append(data)

    def read_new_oxides(self, ):
        NEW_OXIDE = pandas.read_excel(
            self.data_file, sheet_name=self.new_oxide_sheet)
        for row_id in range(NEW_OXIDE.shape[0]):
            oxide = NEW_OXIDE.iloc[row_id, 0]
            ftr = self.get_single_feature_from_str(oxide)
            self.data_new['data_x'].append(ftr)
            self.data_new['names'].append(oxide)
            self.data_new['data_y'].append(0)

        self.data_new['data_x'], self.data_new['mean'], self.data_new['sdv'] = self.__normalize_array(
            self.data_new['data_x'], mean=self.data_featurized[700]['mean'], sdv=self.data_featurized[700]['sdv'])
        self.data_new['size'] = len(self.data_new['data_x'])

    def read_descriptors(self, ):
        A_ATTR = pandas.read_excel(
            self.data_file, sheet_name=self.A_site_sheet)

        for row_index in range(A_ATTR.shape[0]):
            element = A_ATTR.iloc[row_index, 0]
            self.elements_A[element] = {}

            self.elements_A[element]['formal_ox_state'] = A_ATTR.iloc[row_index, 2]
            self.elements_A[element]['radius'] = A_ATTR.iloc[row_index, 3]
            self.elements_A[element]['ele_negativity'] = A_ATTR.iloc[row_index, 4]
            self.elements_A[element]['first_ionization'] = A_ATTR.iloc[row_index, 5]
            self.elements_A[element]['strengths'] = A_ATTR.iloc[row_index, 6]

        B_ATTR = pandas.read_excel(
            self.data_file, sheet_name=self.B_site_sheet)
        for row_index in range(B_ATTR.shape[0]):
            element = str(B_ATTR.iloc[row_index, 2])[
                0]+B_ATTR.iloc[row_index, 0]
            self.elements_B[element] = {}

            self.elements_B[element]['formal_ox_state'] = B_ATTR.iloc[row_index, 2]
            self.elements_B[element]['radius'] = B_ATTR.iloc[row_index, 5]
            self.elements_B[element]['ele_negativity'] = B_ATTR.iloc[row_index, 4]
            self.elements_B[element]['first_ionization'] = B_ATTR.iloc[row_index, 3]
            self.elements_B[element]['strengths'] = B_ATTR.iloc[row_index, 6]

    def __parse_str(self, material):
        fields = []
        cur_field = []

        pre_type = None

        for i, char in enumerate(material):
            if char == 'O':
                fields.append(''.join(cur_field).replace(' ', ''))
                cur_field.clear()
                break
            elif char.isalpha():
                cur_type = 'alpha'
            else:
                cur_type = 'number'

            if cur_type == 'alpha' and len(cur_field) == 2:
                fields.append(''.join(cur_field).replace(' ', ''))
                cur_field.clear()
                cur_field.append(char)
            elif not pre_type or cur_type == pre_type:
                cur_field.append(char)
            else:
                fields.append(''.join(cur_field).replace(' ', ''))
                cur_field.clear()
                cur_field.append(char)
            pre_type = cur_type

        if len(cur_field) > 0:
            fields.append(''.join(cur_field))

        pairs = []

        for i in range(0, len(fields), 2):
            substance = fields[i]
            weight = fields[i+1]
            assert self.__is_float(weight)
            pairs.append((substance, float(weight)))

        total = sum([p[1] for p in pairs])

        if abs(total - 2) > 1e-5:
            return fields, total, pairs

        parsed = [{}, {}]
        total_weight = 0
        index = 0
        for element, weight in pairs:
            if abs(total_weight - 1) < 1e-5:
                index += 1
                total_weight = 0
            total_weight += weight
            parsed[index][element] = weight

        return parsed[0], parsed[1]

    def build_feature(self, add_label=False, convert_log=False):
        '''
        convert raw data to feature vector:
        fv = [0: 'av_2', 1: 'av_3', 2: 'bv_2', 3: 'bv_3', 4: 'bv_4', 5: 'bv_5', 6: 'bv_6', 7: 'r_a', 8: 'neg_a', 9: 'first_ion_a', 10: 'strengths_a', 11: 'r_b', 12: 'neg_b', 13: 'first_ion_b', 14: 'strengths_b', 15: 't', 16:'u']
        '''
        data_map = {}

        for data_id, data in enumerate(self.data_raw):
            data_map[data_id] = data
            e1, e2, rp_lst, is_train = data

            unormalized_feature = self.get_single_feature_from_parsed(e1, e2)

            for temperature in rp_lst:
                if temperature not in self.data_featurized:
                    self.data_featurized[temperature] = {
                        'x': [],
                        'y': [],
                        'name': [],
                        'is_train': []
                    }
                rp = rp_lst[temperature]
                if not self.__is_float(rp) or np.isnan(rp):
                    continue
                rp = float(rp)
                if rp > 5:
                    continue

                self.data_featurized[temperature]['x'].append(
                    unormalized_feature)

                self.data_featurized[temperature]['is_train'].append(
                    is_train[temperature])
                if add_label:
                    self.data_featurized[temperature]['x'][-1].append(data_id)

                if convert_log:
                    self.data_featurized[temperature]['y'].append(
                        np.log(rp)) 
                else:
                    self.data_featurized[temperature]['y'].append(rp)

                self.data_featurized[temperature]['name'].append(
                    self.__dict2str(data[0])+'!@#$%^'+self.__dict2str(data[1]))

        for temperature in self.data_featurized:
            self.data_featurized[temperature]['x'] = np.array(
                self.data_featurized[temperature]['x'])
            self.data_featurized[temperature]['y'] = np.array(
                self.data_featurized[temperature]['y'])
            self.data_featurized[temperature]['size'] = len(
                self.data_featurized[temperature]['x'])

    def get_single_feature_from_parsed(self, ele_A, ele_B, debug=False):
        state_a = [0 for i in range(2)]
        state_b = [0 for i in range(5)]
        r_a = 0
        neg_a = 0
        first_ion_a = 0
        strengths_a = 0
        r_b = 0
        neg_b = 0
        first_ion_b = 0
        strengths_b = 0

        for ele in ele_A:
            weight = ele_A[ele]
            state_a[int(self.elements_A[ele]['formal_ox_state'])-2] += weight
            r_a += self.elements_A[ele]['radius']*weight
            neg_a += self.elements_A[ele]['ele_negativity']*weight
            first_ion_a += self.elements_A[ele]['first_ionization']*weight
            strengths_a += self.elements_A[ele]['strengths']*weight

        e2_new = self.__calculate_ratio(
            siteA=ele_A, siteB=ele_B, A_dict=self.elements_A, B_dict=self.elements_B)
        if not e2_new:
            return None

        for ele in e2_new:
            weight = e2_new[ele]
            state_b[int(self.elements_B[ele]
                        ['formal_ox_state'])-2] += weight
            r_b += self.elements_B[ele]['radius']*weight
            neg_b += self.elements_B[ele]['ele_negativity']*weight
            first_ion_b += self.elements_B[ele]['first_ionization']*weight
            strengths_b += self.elements_B[ele]['strengths']*weight

        t = (r_a+1.35)/math.sqrt(2)/(r_b+1.35)
        u = r_b/1.35

        return state_a+state_b+[r_a, neg_a, first_ion_a, strengths_a, r_b, neg_b, first_ion_b, strengths_b, t, u, ]

    def get_single_feature_from_str(self, material_name, debug=False, normalize=None):
        A_site_elements, B_site_elements = self.__parse_str(material_name)
        unormalized_feature = self.get_single_feature_from_parsed(
            A_site_elements, B_site_elements, debug)
        if normalize:
            return self.__normalize_array([unormalized_feature], mean=self.data_featurized[normalize]['mean'], sdv=self.data_featurized[normalize]['sdv'])[0][0]
        return unormalized_feature

    def get_target(self, mask_dims=[]):
        test_dataset = {'data_x': [], 'data_y': [], 'names': []}
        for oxide, value in [('Sr0.9Cs0.1Co0.9Nb0.1', 0.0101), ('Ba0.4Sr0.4Cs0.2Co0.6Fe0.3Mo0.1', 0.0108), ('Ba0.8Sr0.2Co0.6Fe0.2Nb0.2', 0.0123), ('Ba0.2Sr0.6Pr0.2Co0.6Fe0.3Nb0.1', 0.0150)]:
            test_dataset['data_x'].append(
                self.get_single_feature_from_str(oxide, normalize=700))
            test_dataset['data_y'].append(value)
            test_dataset['names'].append(oxide)

        test_dataset['data_x'] = np.delete(
            test_dataset['data_x'], mask_dims, -1)
        return test_dataset

    @staticmethod
    def shuffle(dataset):
        indices = [i for i in range(dataset['size'])]
        random.shuffle(indices)
        dataset['data_x'] = dataset['data_x'][indices]
        dataset['data_y'] = dataset['data_y'][indices]
        dataset['names'] = dataset['names'][indices]
        return dataset

    @staticmethod
    def change_y(dataset, func):
        if func == 'reverse':
            dataset['data_y'] = 1/dataset['data_y']
        elif func == 'log':
            dataset['data_y'] = np.log10(dataset['data_y'])

    def __calculate_ratio(self, siteB, siteA, A_dict, B_dict):
        unstable_elements = []
        states = []
        cur_state = 6
        for ele in siteA:
            cur_state -= siteA[ele]*A_dict[ele]['formal_ox_state']

        cur_weight = 1
        new_site_B = {}

        for ele in siteB:
            state_lst = self.__get_state_lst(ele, B_dict)
            if not state_lst:
                return
            if len(state_lst) > 1:
                unstable_elements.append(ele)
                states.extend(state_lst)
            else:
                new_site_B[state_lst[0]+ele] = siteB[ele]
                cur_weight -= siteB[ele]
                cur_state -= siteB[ele] * \
                    B_dict[state_lst[0]+ele]['formal_ox_state']

        if len(unstable_elements) == 0:
            return new_site_B

        states = set(states)
        try:
            assert len(states) == 2
        except:
            print(siteA, siteB)
            exit(1)
        states = list(states)
        x1 = (cur_state - int(states[1])*cur_weight) / \
            (int(states[0])-int(states[1]))
        x1 = min(max(x1, 0), cur_weight)
        x2 = cur_weight - x1

        for ue in unstable_elements:
            new_site_B[str(states[0])+ue] = siteB[ue]/cur_weight * x1
            new_site_B[str(states[1])+ue] = siteB[ue]/cur_weight * x2

        return new_site_B

    def denormalize(self, dataset):
        dnew = copy.deepcopy(dataset)

        for i in range(dnew['size']):
            for j in range(len(dnew['data_x'][i])):
                if j not in self.normalize_mask_dims:
                    dnew['data_x'][i][j] = dnew['data_x'][i][j] * \
                        dnew['sdv'][j]+dnew['mean'][j]
        return dnew

    def normalize(self):
        for temperature in self.data_featurized:
            self.data_featurized[temperature]['x'], mean, sdv = self.__normalize_array(
                self.data_featurized[temperature]['x'])
            self.data_featurized[temperature]['mean'] = mean
            self.data_featurized[temperature]['sdv'] = sdv

    def __dict2str(self, dic):
        string = ''
        for key in dic:
            string += key+str(dic[key])
        return string

    def __is_float(self, number):
        try:
            x = float(number)
            return True
        except Exception:
            return False

    def __normalize_array(self, array, mean=None, sdv=None):
        if mean is None and sdv is None:
            mean = np.sum(array, axis=0)/len(array)
            sdv = np.sqrt(np.sum(np.square(array-mean), axis=0)/(len(array)-1))

        new_array = []
        for row in array:
            new_row = []
            for j in range(len(row)):
                if j in self.normalize_mask_dims:
                    new_row.append(row[j])
                else:
                    new_row.append((row[j]-mean[j])/sdv[j])
            new_array.append(new_row)

        return np.array(new_array), mean, sdv

    def split(self, target_temperature, test_ratio=0.2):
        for temperature in self.data_featurized:
            if target_temperature and target_temperature != temperature:
                continue
            test_size = int(
                self.data_featurized[temperature]['size'] * test_ratio)
            is_train = [0 for i in range(
                self.data_featurized[temperature]['size'])]
            random_choice = random.shuffle(
                [i for i in range(self.data_featurized[temperature]['size'])])
            for i in range(test_size):
                is_train[random_choice[i]] = 1

            self.data_featurized[temperature]['is_train'] = is_train

    def get_dataset(self, temperature, split, mask_dims=[], reverse_y=False):
        if split == 'new':
            return {
                'data_x': np.delete(np.array(self.data_new['data_x']), mask_dims, -1),
                'data_y': self.data_new['data_y'],
                'names': self.data_new['names'],
                'mean': np.delete(self.data_new['mean'], mask_dims, -1),
                'sdv': np.delete(np.array(self.data_new['sdv']), mask_dims, -1),
                'size': self.data_new['size']
            }
        elif split == 'full':
            return {
                'data_x': np.delete(np.array(self.data_featurized[temperature]['x']), mask_dims, -1),
                'data_y': np.array(self.data_featurized[temperature]['y']) if not reverse_y else 1/np.array(self.data_featurized[temperature]['y']),
                'names': np.array(self.data_featurized[temperature]['name']),
                'mean': np.delete(np.array(self.data_featurized[temperature]['mean']), mask_dims, -1),
                'sdv': np.delete(np.array(self.data_featurized[temperature]['sdv']), mask_dims, -1),
                'size': len(self.data_featurized[temperature]['x'])
            }
        else:
            data_x = []
            data_y = []
            name = []
            for i in range(self.data_featurized[temperature]['size']):
                if (self.data_featurized[temperature]['is_train'][i] == 1 and split == 'train') or (self.data_featurized[temperature]['is_train'][i] == 0 and split == 'test'):
                    data_x.append(self.data_featurized[temperature]['x'][i])
                    data_y.append(self.data_featurized[temperature]['y'][i])
                    name.append(self.data_featurized[temperature]['name'][i])
            return {
                'data_x': np.delete(np.array(data_x), mask_dims, -1),
                'data_y': np.array(data_y) if not reverse_y else 1. / np.array(data_y),
                'names': np.array(name),
                'mean': np.delete(np.array(self.data_featurized[temperature]['mean']), mask_dims, -1),
                'sdv': np.delete(np.array(self.data_featurized[temperature]['sdv']), mask_dims, -1),
                'size': len(data_x)
            }

    @staticmethod
    def mask_dims(dataset, dims, mask='del'):
        if mask == 'del':
            dataset['data_x'] = np.delete(dataset['data_x'], dims, -1)
        elif mask == 'zero':
            dataset['data_x'][:, dims] = 0
        elif mask == 'random':
            dataset['data_x'][:, dims] = np.random.random(
                (dataset['data_x'].shape[0], len(dims)))
        return dataset

    def knn(self, material, temperature, n):
        feature = material
        if type(material) == str:
            feature = self.get_single_feature_from_str(material)
            feature, _, _ = self.__normalize_array(
                [feature], self.data_featurized[temperature]['mean'], self.data_featurized[temperature]['sdv'])
            feature = feature[0]
        sorted_neighbours = []
        for i, x in enumerate(self.data_featurized[temperature]['x']):
            sorted_neighbours.append([
                self.data_featurized[temperature]['name'][i],
                self.data_featurized[temperature]['x'][i],
                self.data_featurized[temperature]['y'][i],
                self.__cos_similar(x, feature)
            ])

        sorted_neighbours.sort(key=lambda x: x[-1], reverse=True)
        return sorted_neighbours[:n]

    def __cos_similar(self, v1, v2):
        num = float(np.dot(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) 
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

    def __get_state_lst(self, ele, ele_dict):
        state_lst = []
        for key in ele_dict:
            if ele in key:
                state_lst.append(key[0])
        try:
            assert len(state_lst) > 0
        except:
            print('miss ele', ele)
            return None
        return tuple(state_lst)

    @staticmethod
    def split_validation(dataset, prop, idx):
        validation_size = int(dataset['size'] * prop)
        validation_range = [validation_size*idx,
                            min(validation_size*(idx+1), dataset['size'])]
        val_set = {'data_x': [], 'data_y': [], 'names': [], 'size': 0}
        train_set = {'data_x': [], 'data_y': [], 'names': [], 'size': 0}

        for i in range(dataset['size']):
            if validation_range[0] <= i < validation_range[1]:
                val_set['data_x'].append(dataset['data_x'][i])
                val_set['data_y'].append(dataset['data_y'][i])
                val_set['names'].append(dataset['names'][i])
                val_set['size'] += 1
            else:
                train_set['data_x'].append(dataset['data_x'][i])
                train_set['data_y'].append(dataset['data_y'][i])
                train_set['names'].append(dataset['names'][i])
                train_set['size'] += 1
        return train_set, val_set

    @staticmethod
    def loo_validation(dataset, idx):
        val_set = {'data_x': [dataset['data_x'][idx]], 'data_y': [
            dataset['data_y'][idx]], 'names': [dataset['names'][idx]], 'size': 1}
        train_set = copy.deepcopy(dataset)
        train_set['data_x'] = np.delete(train_set['data_x'], idx, axis=0)
        train_set['data_y'] = np.delete(train_set['data_y'], idx, axis=0)
        train_set['names'] = np.delete(train_set['names'], idx, axis=0)
        train_set['size'] = train_set['size'] - 1

        return train_set, val_set

    @staticmethod
    def mask_feature(dataset, masked_idx, pad=None):
        assert max(masked_idx) < len(dataset['data_x'][0])
        data_x = []
        for i in range(dataset['size']):
            x_ = []
            for j in range(len(dataset['data_x'][i])):
                if j in masked_idx and pad is not None:
                    x_.append(pad)
                elif j in masked_idx and pad is None:
                    continue
                else:
                    x_.append(dataset['data_x'][i][j])
            data_x.append(x_)
        dataset['data_x'] = data_x