#!/bin/bash

TIMITROOT=$1
HTKTools=$2

mkdir -p data


MFCCCONF=conf/htk_mfcc.config
FBANKCONF=conf/htk_fbank.config

lower_case=0
upper_case=0
if [ -d $TIMITROOT/TIMIT/TRAIN -a -d $TIMITROOT/TIMIT/TEST ];
 then
   upper_case=1
   TIMIT=$TIMITROOT/TIMIT
elif [ -d $TIMITROOT/timit/train -a -d $TIMITROOT/timit/test ];
 then
   lower_case=1
   TIMIT=$TIMITROOT/timit
else 
   echo "Error: run.sh requires a directory argument (an absolute pathname) that contains TIMIT/TRAIN and TIMIT/TEST or timit/train and timit/test."
   exit 1;
fi

# construct Training, Development, and Testing sets



for DIR in  TEST TRAIN ; do
    dir=`echo $DIR | tr '[:upper:]' '[:lower:]'`
    if [ $lower_case == 1 ] ; then
        DIR=$dir
    fi
    

    (cd $TIMIT ; find $DIR -type d) | sed "s:^:data/timit/:" | tr '[:upper:]' '[:lower:]' | xargs mkdir -p 
    if [ $lower_case == 1 ] ; then
        (cd $TIMIT ; find $DIR -type f -iname S[IX]\*WAV) |  sort | sed "s:^:timit/:" > data/$dir.wav
    else
        (cd $TIMIT ; find $DIR -type f -iname S[IX]\*WAV) |  sort | sed "s:^:TIMIT/:" > data/$dir.wav
    fi
    
    if [ $lower_case == 1 ] ; then
        sed "s/wav$/phn/" data/$dir.wav | sed "s:^:data/:" > data/$dir.phn
        sed "s/wav$/mfc/" data/$dir.wav | sed "s:^:data/:" > data/$dir.scp
	sed "s/wav$/fbank/" data/$dir.wav | sed "s:^:data/:" > data/$dir.fbank
        sed "s/wav$/txt/" data/$dir.wav | sed "s:^:data/:" > data/$dir.txt
    else
        sed "s/WAV$/phn/" data/$dir.wav | tr '[:upper:]' '[:lower:]' | sed "s:^:data/:" > data/$dir.phn
        sed "s/WAV$/mfc/" data/$dir.wav | tr '[:upper:]' '[:lower:]' | sed "s:^:data/:" > data/$dir.scp
        sed "s/WAV$/fbank/" data/$dir.wav | tr '[:upper:]' '[:lower:]' | sed "s:^:data/:" > data/$dir.fbank
        sed "s/WAV$/txt/" data/$dir.wav | tr '[:upper:]' '[:lower:]' | sed "s:^:data/:" > data/$dir.txt
    fi

    paste data/$dir.wav data/$dir.scp | sed "s:^:$TIMITROOT/:" > data/$dir.convert
    $HTKTools/HCopy -C $MFCCCONF -S data/$dir.convert
    paste data/$dir.wav data/$dir.fbank | sed "s:^:$TIMITROOT/:" > data/$dir.convert
    $HTKTools/HCopy -C $FBANKCONF -S data/$dir.convert
    rm -f data/$dir.convert

    if [ $lower_case == 1 ] ; then
        sed "s/.wav$//" data/$dir.wav | while read base ; do
            sed 's/ ax-h$/ axh/' < $TIMITROOT/$base.phn > data/$base.phn
            egrep -v 'h#$' data/$base.phn > data/$base.txt
        done
    else
        sed "s/.WAV$//" data/$dir.wav | while read base ; do
	    outbase=`echo data/$base | tr '[:upper:]' '[:lower:]'`
            sed 's/ ax-h$/ axh/' < $TIMITROOT/$base.PHN > $outbase.phn
            egrep -v 'h#$' $outbase.phn > $outbase.txt
        done
    fi

    # create MLF
    $HTKTools/HLEd -G TIMIT -S data/$dir.phn -i data/${dir}Mono.mlf /dev/null
    rm -f data/$dir.wav

done

# filter the main test set to get the core test set and the development set (the latter being use to test for convergence)
for dset in dev coreTest ; do
    FILTER=`(echo -n '^data/timit/test/dr./(' && cat conf/$dset.spkrs && echo -n ')/s[ix]' )| tr '\n' '|'`
    egrep -i $FILTER data/test.scp > data/$dset.scp
    egrep -i $FILTER data/test.phn > data/$dset.phn
    $HTKTools/HLEd -G TIMIT -S data/$dset.phn -i data/${dset}Mono.mlf /dev/null
done


mkdir -p exp
# create list of monophones
find data/timit/train -name \*phn | xargs cat | awk '{print $3}' | sort -u > exp/monophones

# and derive the dictionary and the modified monophone list from the monophones
egrep -v h# exp/monophones > exp/monophones-h#
paste exp/monophones-h# exp/monophones-h# > exp/dict
echo "!ENTER	[] h#" >> exp/dict
echo "!EXIT	[] h#" >> exp/dict
echo '!ENTER' >> exp/monophones-h#
echo '!EXIT' >> exp/monophones-h#
