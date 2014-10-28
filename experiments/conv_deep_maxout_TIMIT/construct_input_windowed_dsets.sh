python scripts/generate_randomized_hdf5_svm_dataset.py --dset train --seed 0 --n_context_frames 17

python scripts/generate_randomized_hdf5_svm_dataset.py --dset dev --seed 0 --n_context_frames 17

python scripts/generate_randomized_hdf5_svm_dataset.py --dset coreTest --seed 0 --n_context_frames 17

python scripts/generate_rand_binary_svm_dataset.py --dset train --seed 0 --n_context_frames 17 --save_path work/timitCDlabel_fbank_binary_3channel_train.bin
