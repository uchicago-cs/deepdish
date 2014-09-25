bestnumtrimix=$(( `cat bestnumtrimix` - 2))
HTK=$1
PATH=$PATH:$HTK/HTKTools

cd work
cp dict dict2
echo -e "h#\th#" >> dict2

HVite -f -a -T 1 -C hvite.config -H tri-nmix${bestnumtrimix}-npass4/MMF -S train.scp -I trainMono.mlf -i trainFATri.mlf dict2 tiedlist 

awk '{ if ( NF > 3 ) { print $3 } }' trainFATri.mlf | sort | uniq > triphoneLabels


