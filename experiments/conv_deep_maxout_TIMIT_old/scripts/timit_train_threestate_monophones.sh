#
# HTK script to train three state monophones
#

HTKTools=$1
HTKSamples=$2
NMIXMONO=$3
MINTESTMONO=$4
NPASSPERMIX=$5

# template for a prototype model should be in 
# conf/simp.pcf

exp=exp

mkdir -p $exp

# generate a prototype model
cat << "EOF" > exp/proto
~o <VecSize> 39 <MFCC_E_D_A_Z>
~h "proto"
<BeginHMM>
<NumStates> 5
<State> 2
<Mean> 39
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<Variance> 39
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<State> 3
<Mean> 39
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<Variance> 39
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<State> 4
<Mean> 39
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<Variance> 39
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<TransP> 5
0.0 1.0 0.0 0.0 0.0
0.0 0.6 0.4 0.0 0.0
0.0 0.0 0.6 0.4 0.0
0.0 0.0 0.0 0.7 0.3
0.0 0.0 0.0 0.0 0.0
<EndHMM>
EOF

mkdir -p exp/hmm0
$HTKTools/HCompV -T 1 -m -S data/train.scp -M exp/hmm0 -f 0.01  -o proto exp/proto

# now initialize all of the three-state phone models
mkdir -p exp/hmm1
for phn in `cat exp/monophones` ; do
    echo $phn
    $HTKTools/HInit -I data/trainMono.mlf -S data/train.scp \
	-T 1 -M exp/hmm1 -l $phn exp/hmm0/proto
    sed "s/proto/$phn/" exp/hmm1/proto > exp/hmm1/$phn
done

# reestimate with Baum-Welch
mkdir -p exp/hmm2
for phn in `cat exp/monophones` ; do
    echo $phn
    $HTKTools/HRest -I data/trainMono.mlf -S data/train.scp \
	-T 1 -M exp/hmm2 -l $phn exp/hmm1/$phn
done

# combine together to estimate as a single model
# do splitting and re-estimation until likelihood no longer improves



mkdir -p exp/hmm1
