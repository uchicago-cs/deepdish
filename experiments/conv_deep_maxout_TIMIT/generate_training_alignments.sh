bestnumtrimix=$(( `cat bestnumtrimix` - 2))
HTK=$1
PATH=$PATH:$HTK/HTKTools

cd work
cp dict dict2
echo -e "h#\th#" >> dict2

HVite -f -a -T 1 -C hvite.config -H tri-nmix${bestnumtrimix}-npass4/MMF -S train.scp -I trainMono.mlf -i trainFATri.mlf dict2 tiedlist 

HVite -f -a -T 1 -C hvite.config -H tri-nmix${bestnumtrimix}-npass4/MMF -S dev.scp -I devMono.mlf -i devFATri.mlf dict2 tiedlist 

HVite -f -a -T 1 -C hvite.config -H tri-nmix${bestnumtrimix}-npass4/MMF -S coreTest.scp -I coreTestMono.mlf -i coreTestFATri.mlf dict2 tiedlist 


cat trainFATri.mlf devFATri.mlf coreTestFATri.mlf | awk '{ if ( NF > 2 ) { print $3 } }'  | sort | uniq > triphoneLabels


