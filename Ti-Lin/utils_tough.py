#Common utility file
#From Fast-Lin repository
#https://github.com/huanzhang12/CertifiedReLURobustness

import numpy as np
import random
import os
import pandas as pd
from PIL import Image
random.seed(1215)
np.random.seed(1215)


def linf_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def l1_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=1)

def l0_dist(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=0)

def show(img, name = "output.png"):
    """
    Show MNSIT digits in the console.
    """
    np.save('img', img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    # pic.resize((512,512), resample=PIL.Image.BICUBIC)
    pic.save(name)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    return
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def generate_data(data, samples, targeted=True, random_and_least_likely = False, skip_wrong_label = True, start=10,
        target_classes = None, target_type = 0b1111, predictor = None, imagenet=False, remove_background_class=False, save_inputs=False, model_name=None, save_inputs_dir=None):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    ids: true IDs of images in the dataset, if given, will use these images
    target_classes: a list of list of labels for each ids
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    target_candidate_pool = np.eye(data.test_labels.shape[1])
    target_candidate_pool_remove_background_class = np.eye(data.test_labels.shape[1] - 1)
    print('generating labels...')
    total = 0
    #print(ids)
    for i in range(data.test_data.shape[0]):
        if total >= 10:
            break
        predicted_label = -1 # unknown
        original_predict = np.squeeze(predictor(np.array([data.test_data[i]])))
        num_classes = len(original_predict)
        predicted_label = np.argmax(original_predict)
        least_likely_label = np.argmin(original_predict)
        top2_label = np.argsort(original_predict)[-2]
        start_class = 1 if (imagenet and not remove_background_class) else 0
        random_class = predicted_label
        new_seq = [least_likely_label, top2_label, predicted_label]
        while random_class in new_seq:
            random_class = random.randint(start_class, start_class + num_classes - 1)
        new_seq[2] = random_class
        true_label = np.argmax(data.test_labels[start+i])
        seq = []
        print('true_label', true_label)
        print('predicted_label', predicted_label)
        print(start)
        if true_label != predicted_label and skip_wrong_label:
            continue
        else:
            total += 1
            if target_type & 0b10000:
                for c in range(num_classes):
                    if c != predicted_label:
                        seq.append(c)
                        information.append('class'+str(c))
            else:
                if target_type & 0b0100:
                    # least
                    seq.append(new_seq[0])
                    information.append('least')
                if target_type & 0b0001:
                    # top-2
                    seq.append(new_seq[1])
                    information.append('top2')
                if target_type & 0b0010:
                    # random
                    seq.append(new_seq[2])
                    information.append('random')
        print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i, 
            np.argmax(data.test_labels[start+i]), predicted_label, np.argmax(data.test_labels[start+i]) == predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))
        print(seq)
        # print(data.test_data)
        for j in seq:
            # print(data.test_data)
            # skip the original image label
            if (j == np.argmax(data.test_labels[start+i])):
                continue
            inputs.append(data.test_data[start+i])
            if remove_background_class:
                targets.append(target_candidate_pool_remove_background_class[j])
            else:
                targets.append(target_candidate_pool[j])
            true_labels.append(data.test_labels[start+i])
            if remove_background_class:
                true_labels[-1] = true_labels[-1][1:]
            true_ids.append(start+i)
    print('******')
    print(inputs)
    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)
    print('labels generated')
    print('{} images generated in total.'.format(len(inputs)))
    if save_inputs:
        if not os.path.exists(save_inputs_dir):
            os.makedirs(save_inputs_dir)
        save_model_dir = os.path.join(save_inputs_dir,model_name)
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        info_set = list(set(information))
        for info_type in info_set:
            save_type_dir = os.path.join(save_model_dir,info_type)
            if not os.path.exists(save_type_dir):
                os.makedirs(save_type_dir)
            counter = 0
            for i in range(len(information)):
                if information[i] == info_type:
                    df = inputs[i,:,:,0]
                    df = df.flatten()
                    np.savetxt(os.path.join(save_type_dir,'point{}.txt'.format(counter)),df,newline='\t')
                    counter += 1
            target_labels = np.array([np.argmax(targets[i]) for i in range(len(information)) if information[i]==info_type])
            np.savetxt(os.path.join(save_model_dir,model_name+'_target_'+info_type+'.txt'),target_labels,fmt='%d',delimiter='\n') 
    return inputs, targets, true_labels, true_ids, information

