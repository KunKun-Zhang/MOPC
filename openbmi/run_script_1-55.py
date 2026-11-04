"""Train the leave-one-subject-out source ensembles for the 54 OpenBMI target subjects.

For every target subject, cross-subject_cv_by_subject.py pools the prepared data of the other 53 subjects
(both sessions), aligns them with Euclidean alignment and trains five EEGNet models (one per inner fold),
keeping the checkpoint with the best validation accuracy of each fold. Prepare the data first with
OpenBMI/datasets/get_cross_subject_data.py.

    python run_script_1-55.py
    python run_script_1-55.py --seed 2025 --model_save_path /run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject_seed2025/
"""
import os
import sys
import argparse
import subprocess

current_folder_path = os.path.dirname(os.path.abspath(__file__))
run_file = os.path.join(current_folder_path, 'cross-subject_cv_by_subject.py')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the OpenBMI source ensembles')
    parser.add_argument('--seed', type=str, default='2024', help='random seed')
    parser.add_argument('--gpu_id', type=str, default='0', help='CUDA device index')
    parser.add_argument('--model_save_path', type=str, default='/run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject/',
                        help='output directory relative to this folder (format string: model_type, train_samples)')
    args = parser.parse_args()

    for person_id in range(1, 55):
        print(f'Training the source ensemble for target subject {person_id}')
        subprocess.run([
            sys.executable, run_file,
            '--lr', '0.001',
            '--gpu_id', args.gpu_id,
            '--seed', args.seed,
            '--epochs', '1000',
            '--batch_size', '32',
            '--patience', '1000',
            '--train_samples', '1000',
            '--fold', '5',
            '--classes', '2',
            '--person_id', str(person_id),
            '--augmentation', 'none',
            '--model_type', 'EEGNet_features',
            '--model_save_path', args.model_save_path,
        ], check=True)
