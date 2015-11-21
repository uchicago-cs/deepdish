
# Makefile would be better
parallel dot -Tsvg {}.dot -o {}.svg ::: example block forward backward shared

