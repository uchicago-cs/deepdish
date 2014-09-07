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
echo proto > exp/protolist
(cd exp ; echo N | $HTKSamples/HTKDemo/MakeProtoHMMSet ../conf/sim.pcf)


$HTKTools/HCompV -T 1 -m -S data/train.scp -f 0.01 -M . -o new exp/proto

mv new exp/new
mv vFloors exp/vFloors

KFLCFG='-e n en -e aa ao -e ah ax-h -e ah ax -e ih ix -e l el -e sh zh -e uw ux -e er axr -e m em -e n nx -e ng eng -e hh hv -e pau pcl -e pau tcl -e pau kcl -e pau q -e pau bcl -e pau dcl -e pau gcl -e pau epi -e pau h#'

nmix=1
NEWDIR=exp/mono-nmix$nmix-npass0

# concatenate prototype models to build a flat-start model
mkdir -p $NEWDIR
sed '1,3!d' exp/new > $NEWDIR/MMF
cat exp/vFloors >> $NEWDIR/MMF
for i in `cat exp/monophones` ; do
  sed -e "1,3d" -e "s/new/$i/" exp/new >> $NEWDIR/MMF
done

$HTKTools/HLEd -S data/train.txt -i data/trainTxt.mlf /dev/null
$HTKTools/HLStats -T 1 -b exp/bigfn -o -I data/trainTxt.mlf -S data/train.txt exp/monophones-h#
$HTKTools/HBuild -T 1 -n exp/bigfn exp/monophones-h# exp/outLatFile

# HVITE needs to be told to perform cross word context expansion
# cat <<"EOF" > conf/hvite.config
# FORCECXTEXP = TRUE
# ALLOWXWRDEXP = TRUE
# EOF

OPT="-T 1 -m 0 -t 250 150 1000 -S data/train.scp"

echo Start training monophones at: `date`


while [ $nmix -le $NMIXMONO ] ; do

  ## NB the inner loop of both cases is duplicated - change both! 
  if [ $nmix -eq 1 ] ; then
    npass=1;
    while [ $npass -le $NPASSPERMIX ] ; do
      OLDDIR=$NEWDIR
      NEWDIR=exp/mono-nmix$nmix-npass$npass
      mkdir -p $NEWDIR
      $HTKTools/HERest $OPT -I data/trainMono.mlf -H $OLDDIR/MMF -M $NEWDIR exp/monophones > $NEWDIR/LOG
      npass=$(($npass+1))
    done
    echo 'MU 2 {*.state[2-4].mix}' > exp/tmp.hed
    nmix=2
  else
    NEWDIR=exp/mono-nmix$nmix-npass0 
    mkdir -p $NEWDIR
    $HTKTools/HHEd -H $OLDDIR/MMF -M $NEWDIR exp/tmp.hed exp/monophones
    npass=1
    while [ $npass -le $NPASSPERMIX ] ; do
      OLDDIR=$NEWDIR
      NEWDIR=exp/mono-nmix$nmix-npass$npass
      mkdir -p $NEWDIR
      $HTKTools/HERest $OPT -I data/trainMono.mlf -H $OLDDIR/MMF -M $NEWDIR exp/monophones > $NEWDIR/LOG
      npass=$(($npass+1))
    done
    echo 'MU +2 {*.state[2-4].mix}' > exp/tmp.hed
    nmix=$(($nmix+2))
  fi

  # test models
  if [ $nmix -ge $MINTESTMONO ] ; then
    $HTKTools/HVite -t 100 100 4000 -T 1 -H $NEWDIR/MMF -S data/coreTest.scp -i $NEWDIR/recout.mlf -w exp/outLatFile -p 0.0 -s 5.0 exp/dict exp/monophones
    $HTKTools/HResults -T 1 -e '???' h# -I data/coreTestMono.mlf exp/monophones $NEWDIR/recout.mlf
    if $KFLMAP ; then
      $HTKTools/HResults -T 1 -e '???' h# $KFLCFG -I data/coreTestMono.mlf exp/monophones $NEWDIR/recout.mlf
    fi
  fi

done

echo Completed monophone training at: `date`
