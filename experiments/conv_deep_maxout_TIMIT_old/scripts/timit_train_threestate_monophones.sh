#
# HTK script to train three state monophones
#

HTKTools=$1

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
mkdir -p exp/hmm3

# new HMM macro is exp/hmm3/newMacros
$HTKTools/HERest -d exp/hmm2 -M exp/hmm3 -I data/trainMono.mlf -S data/train.scp exp/monophones

for i in `seq 1 1 12` ; do
    $HTKTools/HERest -H exp/hmm3/newMacros  -M exp/hmm3 \
	-I data/trainMono.mlf -S data/train.scp -T 1 \
	exp/monophones
done

# create a grammar
(echo -n '$phone = ' && cat exp/monophones && echo -n ' ;' )| sed ':a;N;$!ba;s/\n/ | /g' | sed 's:|  ;:;:' > exp/gram
echo '' >> exp/gram
echo '( <$phone> )' >> exp/gram

$HTKTools/HParse  exp/gram monophonewrdnet
mv monophonewrdnet exp/

paste exp/monophones exp/monophones > exp/dict

$HTKTools/HVite -H exp/hmm3/newMacros -S data/dev.scp \
    -i exp/hmm3/recout.mlf -w exp/monophonewrdnet \
    -p -6.0 -s 5.0 exp/dict exp/monophones

accuracy=`$HTKTools/HResults -I data/devMono.mlf exp/monophones exp/hmm3/recout.mlf | grep '^WORD:' | awk '{ print $3 }' | sed 's:Acc=::'`

echo $accuracy > exp/hmm3/dev_accuracy

$HTKTools/HVite -H exp/hmm3/newMacros -S data/train.scp \
    -i exp/hmm3/train_recout.mlf -w exp/monophonewrdnet \
    -p -6.0 -s 5.0 exp/dict exp/monophones

$HTKTools/HResults -I data/trainMono.mlf exp/monophones exp/hmm3/train_recout.mlf | grep '^WORD:' | awk '{ print $3 }' | sed 's:Acc=::' > exp/hmm3/train_accuracy
