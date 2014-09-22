TIMIT=/home/mark/Data/timit
MFCCCONFIG=conf/htk_mfcc.config
HTK=/home/mark/Software/htk
HTKSamples=$(HTK)/samples
CANTAB=HTKtimit.sh
NMIXMONO=30              # maximum number of Gaussians per state in monophones
NMIXTRI=30               # maximum number of Gaussians per state in triphones
MINTESTMONO=1            # test the monophones after this number of Gaussians
MINTESTTRI=1             # test the triophones after this number of Gaussians
NPASSPERMIX=8            # number of fwd/bwd passes per mixture incr

all: bestnumtrimix

bestnumtrimix:
	bash $(CANTAB) $(TIMIT) $(HTK) $(HTKSamples) $(NMIXMONO) $(NMIXTRI) $(NPASSPERMIX)

.PHONY: clean
clean:
	rm -rf work bestnumtrimix
