#!/bin/bash -ex

# title notice
echo "HTK training for TIMIT from Cantab Research"
echo "See http://www.cantabResearch.com/HTKtimit.html"

# (C) Cantab Research: Permission is granted for free distribution and
# usage provided the copyright notice and the title notice are not
# altered.
#
# version 1.0: 22Jun06 - first public release
# version 1.1: 03Jul06 - vFloors was calculated but not used (Andrew Morris)
#                      - changed name from MODEL to MMF
#                      - fixed spurious HERest WARNING [-2331]
# version 1.2: 23Aug06 - fixed insertion errors caused by h# in references
# version 1.3: 14Sep06 - fixed typo KFLMAP -> KFLCFG in HResults line
# version 1.4: 26Oct13 - fixed minor bug - penultimate model was used in HHEd
#
# based on the Matlab scipts by Dan Ellis

# you *must* edit this block to reflect your local setup
TIMIT=$1
HTK=$2
SAMPLES=$3
PATH=$PATH:$HTK/HTKTools

# do all work in this sub directory
WORK=work
mkdir -p $WORK

# uncomment this line to log everything in the working directory
# exec >& $WORK/$0.log

# some options to play with once the base system is working
HTKMFCC=true             # use HTK's MFCC instead of external USER executable
NMIXMONO=$4              # maximum number of Gaussians per state in monophones
NMIXTRI=$5               # maximum number of Gaussians per state in triphones
MINTESTMONO=1            # test the monophones after this number of Gaussians
MINTESTTRI=1             # test the triophones after this number of Gaussians
NPASSPERMIX=$6            # number of fwd/bwd passes per mixture increase
TESTSET=dev         # set to "test" for full test set or "coreTest"
KFLMAP=false             # set to true to addionally output KFL mapped scores
TRITHRESH=$7              # Threshold to use for triphone clustering

echo "Start at `date`"

## keep a copy of this script,  plp source and executable
cp -p $0  $WORK

cd $WORK

pwd
echo $WORK
echo $HTK
echo $SAMPLES

# write the timit config used if using HTK MFCC
if $HTKMFCC ; then
  cat <<"EOF" > config
SOURCEKIND     = WAVEFORM
SOURCEFORMAT   = NIST
SAVECOMPRESSED = TRUE
SAVEWITHCRC    = TRUE
TARGETKIND     = MFCC_E_D_A_Z
TARGETRATE     = 100000
SOURCERATE     = 625
WINDOWSIZE     = 250000.0
PREEMCOEF      = 0.97
ZMEANSOURCE    = TRUE
USEHAMMING     = TRUE
CEPLIFTER      = 22
NUMCHANS       = 26
NUMCEPS        = 12
ENORMALISE     = TRUE
ESCALE         = 1.0
EOF
else
  cat <<"EOF" > config
HPARM: SOURCEFORMAT = HTK   
# HPARM: SOURCEKIND = USER
# HPARM: TARGETKIND = USER_D_A
EOF
fi



# read the TIMIT disk and encode into acoutic features
for DIR in train test ; do
  # create a mirror of the TIMIT directory structure
  (cd $TIMIT ; find $DIR -type d) | xargs mkdir -p

  # generate lists of files
  (cd $TIMIT ; find $DIR -type f -name s[ix]\*wav) | sort > $DIR.wav
  sed "s/wav$/phn/" $DIR.wav > $DIR.phn
  sed "s/wav$/mfc/" $DIR.wav > $DIR.scp
  sed "s/wav$/txt/" $DIR.wav > $DIR.txt

  # generate the acoutic feature vectors
  if $HTKMFCC ; then
    paste $DIR.wav $DIR.scp | sed "s:^:$TIMIT/:" > $DIR.convert
    HCopy -C config -S $DIR.convert
    rm -f $DIR.convert
  else
    sed "s/.wav$//" $DIR.wav | while read base ; do
      wavAddHead -skip 1024 $TIMIT/$base.wav tmp.wav
      plp -outputHTK tmp.wav tmp.plp
      deltas $* tmp.plp $base.mfc
    done
    rm -f tmp.wav
  fi

  # ax-h conflicts with HTK's triphone naming convention, so change it
  # also generate .txt files suitable for use in language modelling
  sed "s/.wav$//" $DIR.wav | while read base ; do
    sed 's/ ax-h$/ axh/' < $TIMIT/$base.phn > $base.phn
    egrep -v 'h#$' $base.phn > $base.txt
  done

  # create MLF
  HLEd -S $DIR.phn -i ${DIR}Mono.mlf /dev/null

  rm -f $DIR.wav
done

# filter the main test set to get the core test set
FILTER='^test/dr./[mf](DAB0|WBT0|ELC0|TAS1|WEW0|PAS0|JMP0|LNT0|PKT0|LLL0|TLS0|JLM0|BPM0|KLT0|NLP0|CMJ0|JDH0|MGD0|GRT0|NJM0|DHC0|JLN0|PAM0|MLD0)/s[ix]'
# core test is actually the development set
# FILTER=`(echo -n '^test/dr./(' && cat ../conf/dev.spkrs && echo -n ')/s[ix]' )| tr '\n' '|'`
# echo "FILTER = $FILTER"
egrep -i $FILTER test.scp > coreTest.scp
egrep -i $FILTER test.phn > coreTest.phn
HLEd -S coreTest.phn -i coreTestMono.mlf /dev/null


FILTER=`(echo -n '^test/dr./(' && cat ../conf/dev.spkrs && echo -n ')/s[ix]' )| tr '\n' '|'`
echo "FILTER = $FILTER"
egrep -i $FILTER test.scp > dev.scp
egrep -i $FILTER test.phn > dev.phn
HLEd -S dev.phn -i devMono.mlf /dev/null

# create list of monophones
find train -name \*phn | xargs cat | awk '{print $3}' | sort -u > monophones

# and derive the dictionary and the modified monophone list from the monophones
egrep -v h# monophones > monophones-h#
paste monophones-h# monophones-h# > dict
echo "!ENTER	[] h#" >> dict
echo "!EXIT	[] h#" >> dict
echo '!ENTER' >> monophones-h#
echo '!EXIT' >> monophones-h#


# generate a template for a prototype model
cat <<"EOF" > sim.pcf
<BEGINproto_config_file>
<COMMENT>
   This PCF produces a single mixture, single stream prototype system
<BEGINsys_setup>
hsKind: P
covKind: D
nStates: 3
nStreams: 1
sWidths: 39
mixes: 1
parmKind: MFCC_E_D_A_Z
vecSize: 39
outDir: .
hmmList: protolist
<ENDsys_setup>
<ENDproto_config_file>
EOF

if ! $HTKMFCC ; then
  sed 's/^parmKind: MFCC_E_D_A_Z$/parmKind: USER_D_A/' < sim.pcf > tmp.pcf
  mv tmp.pcf sim.pcf
fi

# generate a prototype model
echo proto > protolist
echo N | $SAMPLES/HTKDemo/MakeProtoHMMSet sim.pcf

if ! $HTKMFCC ; then
  CONFIG="-C config"
fi

HCompV $CONFIG -T 1 -m -S train.scp -f 0.01 -M . -o new proto

KFLCFG='-e n en -e aa ao -e ah ax-h -e ah ax -e ih ix -e l el -e sh zh -e uw ux -e er axr -e m em -e n nx -e ng eng -e hh hv -e pau pcl -e pau tcl -e pau kcl -e pau q -e pau bcl -e pau dcl -e pau gcl -e pau epi -e pau h#'

nmix=1
NEWDIR=mono-nmix$nmix-npass0

# concatenate prototype models to build a flat-start model
mkdir -p $NEWDIR
sed '1,3!d' new > $NEWDIR/MMF
cat vFloors >> $NEWDIR/MMF
for i in `cat monophones` ; do
  sed -e "1,3d" -e "s/new/$i/" new >> $NEWDIR/MMF
done

HLEd -S train.txt -i trainTxt.mlf /dev/null
HLStats -T 1 -b bigfn -o -I trainTxt.mlf -S train.txt monophones-h#
HBuild -T 1 -n bigfn monophones-h# outLatFile

# HVITE needs to be told to perform cross word context expansion
cat <<"EOF" > hvite.config
FORCECXTEXP = TRUE
ALLOWXWRDEXP = TRUE
EOF

OPT="$CONFIG -T 1 -m 0 -t 250 150 1000 -S train.scp"

echo Start training monophones at: `date`


while [ $nmix -le $NMIXMONO ] ; do

  ## NB the inner loop of both cases is duplicated - change both! 
  if [ $nmix -eq 1 ] ; then
    npass=1;
    while [ $npass -le $NPASSPERMIX ] ; do
      OLDDIR=$NEWDIR
      NEWDIR=mono-nmix$nmix-npass$npass
      mkdir -p $NEWDIR
      HERest $OPT -I trainMono.mlf -H $OLDDIR/MMF -M $NEWDIR monophones > $NEWDIR/LOG
      npass=$(($npass+1))
    done
    echo 'MU 2 {*.state[2-4].mix}' > tmp.hed
    nmix=2
  else
    OLDDIR=$NEWDIR
    NEWDIR=mono-nmix$nmix-npass0 
    mkdir -p $NEWDIR
    HHEd -H $OLDDIR/MMF -M $NEWDIR tmp.hed monophones
    npass=1
    while [ $npass -le $NPASSPERMIX ] ; do
      OLDDIR=$NEWDIR
      NEWDIR=mono-nmix$nmix-npass$npass
      mkdir -p $NEWDIR
      HERest $OPT -I trainMono.mlf -H $OLDDIR/MMF -M $NEWDIR monophones > $NEWDIR/LOG
      npass=$(($npass+1))
    done
    echo 'MU +2 {*.state[2-4].mix}' > tmp.hed
    nmix=$(($nmix+2))
  fi

  # test models
  if [ $nmix -ge $MINTESTMONO ] ; then
    HVite $CONFIG -t 100 100 4000 -T 1 -H $NEWDIR/MMF -S $TESTSET.scp -i $NEWDIR/recout.mlf -w outLatFile -p 0.0 -s 5.0 dict monophones
    HResults -T 1 -e '???' h# -I ${TESTSET}Mono.mlf monophones $NEWDIR/recout.mlf
    if $KFLMAP ; then
      HResults -T 1 -e '???' h# $KFLCFG -I ${TESTSET}Mono.mlf monophones $NEWDIR/recout.mlf
    fi
  fi

done

echo Completed monophone training at: `date`


# generate the list of seen triphones and trainTri.mlf
echo "TC" > mktri.led
# HLEd -n trainTriphones -l '*' -i trainTri.mlf mktri.led trainMono.mlf
# why are we including the -l '*' won't that mess up finding the correct files?????
HLEd -n trainTriphones -i trainTri.mlf mktri.led trainMono.mlf



# and generate all possible triphones models
perl -e '
while($phone = <>) {
  chomp $phone;
  push @plist, $phone;
}

print "h#\n";
for($i = 0; $i < scalar(@plist); $i++) {
  for($j = 0; $j < scalar(@plist); $j++) {
    if($plist[$j] ne "h#") {
      for($k = 0; $k < scalar(@plist); $k++) {
	print "$plist[$i]-$plist[$j]+$plist[$k]\n";
      }
    }
  }
}' < monophones > allTriphones

# generate mktri.hed
$SAMPLES/HTKTutorial/maketrihed monophones trainTriphones

# convert the single Gaussian model to triphones
OLDDIR=mono-nmix1-npass$NPASSPERMIX
NEWDIR=tri-nmix1-npass0a
mkdir -p $NEWDIR
HHEd -H $OLDDIR/MMF -M $NEWDIR mktri.hed monophones 

# reestimate all seen triphones independently
OLDDIR=$NEWDIR
NEWDIR=tri-nmix1-npass0b
mkdir -p $NEWDIR
HERest $OPT -I trainTri.mlf -H $OLDDIR/MMF -M $NEWDIR -s $NEWDIR/stats trainTriphones > $NEWDIR/LOG

# build the question set for tree based state clustering
cat <<EOF > tree.hed
RO 200 tri-nmix1-npass0b/stats
TR 0

QS "L_Stop" {b-*, d-*, g-*, p-*, t-*, k-*, dx-*, q-*}
QS "R_Stop" {*+b, *+d, *+g, *+p, *+t, *+k, *+dx, *+q}
QS "L_Affricate" {jh-*, ch-*}
QS "R_Affricate" {*+jh, *+ch}
QS "L_Fricative" {s-*, sh-*, z-*, zh-*, f-*, th-*, v-*, dh-*}
QS "R_Fricative" {*+s, *+sh, *+z, *+zh, *+f, *+th, *+v, *+dh}
QS "L_Nasal" {m-*, n-*, ng-*, em-*, en-*, eng-*, nx-*}
QS "R_Nasal" {*+m, *+n, *+ng, *+em, *+en, *+eng, *+nx}
QS "L_SemivowelGlide" {l-*, r-*, w-*, y-*, hh-*, hv-*, el-*}
QS "R_SemivowelGlide" {*+l, *+r,, *+w, *+y, *+hh, *+hv, *+el}
QS "L_Vowel" {iy-*, ih-*, eh-*, ey-*, ae-*, aa-*, aw-*, ay-*, ah-*, ao-*, oy-*, ow-*, uh-*, uw-*, ux-*, er-*, ax-*, ix-*, axr-*, axh-*}
QS "R_Vowel" {*+iy, *+ih, *+eh, *+ey, *+ae, *+aa, *+aw, *+ay, *+ah, *+ao, *+oy, *+ow, *+uh, *+uw, *+ux, *+er, *+ax, *+ix, *+axr, *+axh}
QS "L_Other" {pau-*, epi-*, h#-*}
QS "R_Other" {*+pau, *+epi, -*h#}


EOF

# and add in the single phone questions and tie the transition matrices
perl -e '
while($phone = <>) {
  chomp $phone;
  push @plist, $phone;
}

for($i = 0; $i < scalar(@plist); $i++) {
  print "QS \"L_$plist[$i]\" {$plist[$i]-*}\n";
  print "QS \"R_$plist[$i]\" {*+$plist[$i]}\n";
}

print "\nTR 2\n";
for($i = 0; $i < scalar(@plist); $i++) {
  for($j = 2; $j < 5; $j++) {
    print "TB TRITHRESH \"ST_$plist[$i]_s${j}\" {(\"$plist[$i]\", \"*-$plist[$i]+*\", \"$plist[$i]+*\", \"*-$plist[$i]\").state[$j]}\n";
  }
}

print "\nTR 1\n";
print "AU \"allTriphones\"\n";
print "CO \"tiedlist\"\n";
print "ST \"trees\"\n";
' < monophones | sed "s:TRITHRESH:$TRITHRESH:" >> tree.hed

# now perform topdown tree based clustering
OLDDIR=$NEWDIR
NEWDIR=tri-nmix1-npass0c
mkdir -p $NEWDIR
HHEd -B -H $OLDDIR/MMF -M $NEWDIR tree.hed trainTriphones

# reestimate and mix up the state clustered triphones
nmix=1
bestaccuracy="-1"
while [ $nmix -le $NMIXTRI ] ; do

  ## NB the inner loop of both cases is duplicated - change both! 
  if [ $nmix -eq 1 ] ; then
    npass=1;
    while [ $npass -le $NPASSPERMIX ] ; do
      OLDDIR=$NEWDIR
      NEWDIR=tri-nmix$nmix-npass$npass
      mkdir -p $NEWDIR
      HERest $OPT -I trainTri.mlf -H $OLDDIR/MMF -M $NEWDIR tiedlist > $NEWDIR/LOG
      npass=$(($npass+1))
    done
    echo 'MU 2 {*.state[2-4].mix}' > tmp.hed
    nmix=2
  else
    OLDDIR=$NEWDIR
    NEWDIR=tri-nmix$nmix-npass0 
    mkdir -p $NEWDIR
    HHEd -H $OLDDIR/MMF -M $NEWDIR tmp.hed tiedlist
    npass=1
    while [ $npass -le $NPASSPERMIX ] ; do
      OLDDIR=$NEWDIR
      NEWDIR=tri-nmix$nmix-npass$npass
      mkdir -p $NEWDIR
      HERest $OPT -I trainTri.mlf -H $OLDDIR/MMF -M $NEWDIR tiedlist > $NEWDIR/LOG
      npass=$(($npass+1))
    done
    echo 'MU +2 {*.state[2-4].mix}' > tmp.hed
    nmix=$(($nmix+2))
  fi

  # test models
  if [ $nmix -ge $MINTESTTRI ] ; then
    HVite $CONFIG -t 100 100 4000 -T 1 -C hvite.config -H $NEWDIR/MMF -S $TESTSET.scp -i $NEWDIR/recout.mlf -w outLatFile -p 0.0 -s 5.0 dict tiedlist 
    HResults -T 1 -e '???' h# -I ${TESTSET}Mono.mlf monophones $NEWDIR/recout.mlf | grep '^WORD:' | awk '{ print $3 }' | sed 's:Acc=::' > $NEWDIR/accuracy
    curaccuracy=`cat $NEWDIR/accuracy`
    isbigger=`echo "$bestaccuracy < $curaccuracy" | bc`
    if [ $isbigger -gt 0 ] ; then
	bestnumtrimix=$nmix
	bestaccuracy=$curaccuracy
    fi
    if $KFLMAP ; then
      HResults -T 1 -e '???' h# $KFLCFG -I ${TESTSET}Mono.mlf monophones $NEWDIR/recout.mlf
    fi
  fi

done

echo $bestnumtrimix > ../bestnumtrimix

# and exit happy
exit 0
