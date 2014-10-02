TIMIT=/home/mark/Data/timit
MFCCCONFIG=conf/htk_mfcc.config
FBANKCONFIG=conf/htk_fbank.config
HTK=/home/mark/Software/htk
HTKSamples=$(HTK)/samples
CANTAB=HTKtimit.sh
NMIXMONO=56              # maximum number of Gaussians per state in monophones
NMIXTRI=56               # maximum number of Gaussians per state in triphones
MINTESTMONO=1            # test the monophones after this number of Gaussians
MINTESTTRI=1             # test the triophones after this number of Gaussians
NPASSPERMIX=4            # number of fwd/bwd passes per mixture incr
TRITHRESH=2500           # Threshold to use for triphone clustering

all: bestnumtrimix work/train.fbank work/triphoneLabels

bestnumtrimix:
	bash -ex $(CANTAB) $(TIMIT) $(HTK) $(HTKSamples) $(NMIXMONO) $(NMIXTRI) $(NPASSPERMIX) $(TRITHRESH)

work/triphoneLabels:
	bash -ex generate_training_alignments.sh $(HTK)


work/timit_CDlabel_fbank_train.hdf5:
	bash -ex make_timit_hdf5.sh $(HTK)/HTKTools $(TIMIT) $(FBANKCONFIG)

vis_greasy:
	bash extract_greasy_pictures.sh $(TIMIT)/.. $(HTK)/HTKTools


.PHONY: clean
clean:
	rm -rf work bestnumtrimix vis_greasy
