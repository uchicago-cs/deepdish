TIMIT=/home/mark/Data
MFCCCONFIG=conf/htk_mfcc.config
HTK=/home/mark/Software/htk
HTKSamples=$(HTK)/samples
NMIXMONO=20              # maximum number of Gaussians per state in monophones
NMIXTRI=20               # maximum number of Gaussians per state in triphones
MINTESTMONO=1            # test the monophones after this number of Gaussians
MINTESTTRI=1             # test the triophones after this number of Gaussians
NPASSPERMIX=4            # number of fwd/bwd passes per mixture incr

all: dataprep trainmonophone

trainmonophone: data
	echo $(HTKSamples); echo $(HTK);
	bash scripts/timit_train_monophone.sh $(HTK)/HTKTools $(HTKSamples) $(NMIXMONO) $(MINTESTMONO) $(NPASSPERMIX)

dataprep:
	bash scripts/timit_data_prep.sh $(TIMIT) $(HTK)/HTKTools



clean:
	rm -rf data exp
