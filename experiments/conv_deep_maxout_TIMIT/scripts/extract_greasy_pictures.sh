# pick the utterance



TIMITROOT=$1
HTKTools=$2

mkdir -p vis_greasy

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
   echo "lower case TIMIT paths"
else 
   echo "Error: run.sh requires a directory argument (an absolute pathname) that contains TIMIT/TRAIN and TIMIT/TEST or timit/train and timit/test."
   exit 1;
fi



if [ $lower_case == 1 ] ; then
  utt=$TIMIT/train/dr1/fcjf0/sa1
  wrd=$utt.wrd
  phn=$utt.phn
  wav=$utt.wav
else
  utt=`echo train/dr1/fcjf0/sa1 | tr '[:lower:]' '[:upper:]'`
  utt=$TIMIT/$utt
  wrd=$utt.WRD
  phn=$utt.PHN
  wav=$utt.WAV
fi

fbank=vis_greasy/sa1.fbank


greasystart=`grep greasy $wrd | awk '{ print $1 }'`
greasyend=`grep greasy $wrd | awk '{ print $2 }'`

greasystartfbank=$(( $greasystart / 160 ))
greasyendfbank=$(( $greasyend / 160 ))
nframes=$(( 3 * (greasyendfbank - greasystartfbank + 1) ))

$HTKTools/HList -C $FBANKCONF  -s $greasystartfbank -e $greasyendfbank -i 41 $wav | head -n $(( nframes +1 )) | tail  -n $nframes | awk '(NR-1) % 3 == 0' | awk '{ for (i=2; i<NF-1; i++) printf $i " "; print $(NF-1)}'  > vis_greasy/greasy.dat

# convert data file to png jet
python scripts/matrix_to_rgb_jet.py vis_greasy/greasy.dat vis_greasy/greasy.png --colormap jet --resize_factor 6.0

# extract the waveform a quarter of the way in
greasyquartstart=$(( $greasystart + ($greasyend - $greasystart)/4))
# using 16 by 25 millisecond windows
greasyquartend=$(( $greasyquartstart + 399 ))
nlines=20
$HTKTools/HList -F NIST -s $greasyquartstart -e $greasyquartend -i 20 $wav | head -n 21 | tail -n 20 | awk '{ for (i=2; i<NF; i++) printf $i " "; print $(NF)}'

echo  "\\documentclass{standalone}

\\usepackage{tikz}
\\usepackage{pgfplots}
\\usepackage{verbatim}

\\begin{document}
\\begin{tikzpicture}
    \\begin{axis}[
        xlabel=Time \$(ms)\$,
        ylabel=Amplitude]
    \\addplot[smooth,mark=*,blue] plot coordinates {
" > vis_greasy/greasy_wave.tex

$HTKTools/HList -F NIST -s $greasyquartstart -e $greasyquartend -i 20 $wav | head -n 21 | tail -n 20 | awk '{ for (i=2; i<NF; i++) printf $i " "; print $(NF)}' | tr " " "\n" | paste --delimiters=', ' <(seq `bc -l <<< $greasyquartstart/16` `bc -l <<< 1/16` `bc -l <<< $greasyquartend/16`) - | sed "s:^:(:" | sed "s:$:):" >> vis_greasy/greasy_wave.tex

echo "    };

    \\end{axis}
\\end{tikzpicture}
\\end{document}
" >> vis_greasy/greasy_wave.tex

pdflatex -output-directory vis_greasy vis_greasy/greasy_wave.tex 
convert -density 300 vis_greasy/greasy_wave.pdf -quality 90 vis_greasy/greasy_wave.png


fbank_cmels="69.2692 138.538 207.808 277.077 346.346 415.615 484.885 554.154 623.423 692.692 761.961 831.231 900.5 969.769 1039.04 1108.31 1177.58 1246.85 1316.12 1385.38 1454.65 1523.92 1593.19 1662.46 1731.73 1801 1870.27 1939.54 2008.81 2078.08 2147.35 2216.61 2285.88 2355.15 2424.42 2493.69 2562.96 2632.23 2701.5 2770.77"

greasyquartstartfbank=$((greasyquartstart/160))

echo  "\\documentclass{standalone}

\\usepackage{tikz}
\\usepackage{pgfplots}
\\usepackage{verbatim}

\\begin{document}
\\begin{tikzpicture}
    \\begin{axis}[
        xlabel=Mels,
        ylabel=Log Energy]
    \\addplot[smooth,mark=*,red] plot coordinates {
" > vis_greasy/greasy_fbank.tex


$HTKTools/HList -C $FBANKCONF  -s $greasyquartstartfbank -e $(( $greasyquartstartfbank )) -i 41 $wav | head -n 2 | tail  -n 1 | awk '{ for (i=2; i<NF-1; i++) printf $i " "; print $(NF-1)}' | tr " " "\n" | paste --delimiters=', ' <(echo ${fbank_cmels} | tr " " "\n") - | sed "s:^:(:" | sed "s:$:):"  >> vis_greasy/greasy_fbank.tex

echo "    };

    \\end{axis}
\\end{tikzpicture}
\\end{document}
" >> vis_greasy/greasy_fbank.tex

pdflatex -output-directory vis_greasy vis_greasy/greasy_fbank.tex 
convert -density 300 vis_greasy/greasy_fbank.pdf -quality 90 vis_greasy/greasy_fbank.png

$HTKTools/HList -C $FBANKCONF  -s $greasyquartstartfbank -e $(( $greasyquartstartfbank )) -i 41 $wav | head -n 2 | tail  -n 1 | awk '{ for (i=2; i<NF-1; i++) printf $i " "; print $(NF-1)}' | tr " " "\n" > vis_greasy/greasy_fbank_column.dat

python scripts/matrix_to_rgb_jet.py vis_greasy/greasy_fbank_column.dat vis_greasy/greasy_fbank_column.png --colormap jet --resize_factor 6.0
