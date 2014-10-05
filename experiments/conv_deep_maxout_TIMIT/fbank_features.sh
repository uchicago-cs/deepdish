TIMIT=$1
HTK=$2
PATH=$PATH:$HTK/HTKTools
WORK=work

cd $WORK

cat <<"EOF" > fbankconfig
SOURCEKIND     = WAVEFORM
NATURALWRITEORDER = T
SOURCEFORMAT = NIST
SAVECOMPRESSED = FALSE
SAVEWITHCRC    = FALSE
TARGETKIND     = FBANK_E_D_A
TARGETRATE     = 100000.0
WINDOWSIZE     = 250000.0
PREEMCOEF      = 0.97
ZMEANSOURCE    = TRUE
USEHAMMING     = TRUE
NUMCHANS       = 40
ENORMALISE     = TRUE
ESCALE         = 1.0
SOURCERATE     = 625

EOF

# read the TIMIT disk and encode into acoutic features
for DIR in train test ; do
  sed "s/txt$/fbank/" $DIR.txt > $DIR.fbank
  sed "s/txt$/wav/" $DIR.txt > $DIR.wav


  paste $DIR.wav $DIR.fbank | sed "s:^:$TIMIT/:" > $DIR.convert
  HCopy -C fbankconfig -S $DIR.convert
  rm -f $DIR.convert

done

# filter the main test set to get the core test set
FILTER='^test/dr./[mf](DAB0|WBT0|ELC0|TAS1|WEW0|PAS0|JMP0|LNT0|PKT0|LLL0|TLS0|JLM0|BPM0|KLT0|NLP0|CMJ0|JDH0|MGD0|GRT0|NJM0|DHC0|JLN0|PAM0|MLD0)/s[ix]'
# core test is actually the development set
# FILTER=`(echo -n '^test/dr./(' && cat ../conf/dev.spkrs && echo -n ')/s[ix]' )| tr '\n' '|'`
# echo "FILTER = $FILTER"
egrep -i $FILTER test.fbank > coreTest.fbank


FILTER=`(echo -n '^test/dr./(' && cat ../conf/dev.spkrs && echo -n ')/s[ix]' )| tr '\n' '|'`
echo "FILTER = $FILTER"
egrep -i $FILTER test.fbank > dev.fbank

exit 0
