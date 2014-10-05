TIMIT=/home/mark/Data
MFCCCONFIG=conf/htk_mfcc.config
HTK=/home/mark/Software/htk
HTKSamples=$(HTK)/samples
NMIXMONO=20              # maximum number of Gaussians per state in monophones
NMIXTRI=20               # maximum number of Gaussians per state in triphones
MINTESTMONO=1            # test the monophones after this number of Gaussians
MINTESTTRI=1             # test the triophones after this number of Gaussians
NPASSPERMIX=4            # number of fwd/bwd passes per mixture incr

all: dataprep trainmonophone triphonetrees traintriphone

.PHONY: traintriphone
traintriphone: exp/triphones
	bash scripts/timit_train_triphone.sh $(HTK)/HTKTools $(NMIXTRI) $(MINTESTTRI) $(NPASSPERMIX)

.PHONY: triphonetrees
triphonetrees: exp
	bash scripts/triphone_trees.sh $(HTK)/HTKTools $(HTKSamples) $(NPASSPERMIX)

.PHONY: trainmonophone
trainmonophone: data
	bash scripts/timit_train_monophone.sh $(HTK)/HTKTools

.PHONY: dataprep
dataprep:
	bash scripts/timit_data_prep.sh $(TIMIT) $(HTK)/HTKTools


vis_greasy: data
	bash scripts/extract_greasy_pictures.sh $(TIMIT) $(HTK)/HTKTools

.PHONY: clean
clean:
	rm -rf data exp vis_greasy
