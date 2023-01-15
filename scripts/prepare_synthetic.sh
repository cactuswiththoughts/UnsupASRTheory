#!/usr/bin/env zsh

target_dir=$1
min_phones=$2
function error
{
    if [ -z "$1" ]
    then
        message="fatal error"
    else
        message="fatal error: $1"
    fi

    echo $message
    echo "finished at $(date)"
    exit 1
}

stage=1
stop_stage=7
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/all.wrd --only-source --destdir $target_dir --thresholdsrc 0 --padding-factor 1 --dict-only
    cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' >! $target_dir/words.txt
    rm $target_dir/dict.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/all.phn --only-source --destdir $target_dir --thresholdsrc 0 --padding-factor 1 --dict-only
    python scripts/wrd_to_char.py ${target_dir}/words.txt ${target_dir}/phones.txt 
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    paste $target_dir/words.txt $target_dir/phones.txt >! $target_dir/lexicon.lst

    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc $min_phones --padding-factor 1 --dict-only

    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst >! $target_dir/lexicon_filtered.lst
fi
  
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    cp $target_dir/all.wrd $target_dir/lm.upper.lid.txt
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.0 --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt >! $target_dir/phones/lm.phones.filtered.txt
    cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
    echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones/lm.phones.filtered.txt --workers 70 --only-source --destdir $target_dir/phones --srcdict $target_dir/phones/dict.phn.txt
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    $KENLM_ROOT/lmplz -o 2 < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 >! $target_dir/kenlm.wrd.o200.arpa
    $KENLM_ROOT/build_binary $target_dir/kenlm.wrd.o200.arpa $target_dir/kenlm.wrd.o200.bin
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o200.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
    python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o200.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    $KENLM_ROOT/lmplz -o 2 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback >! $target_dir/phones/lm.phones.filtered.02.arpa
    $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.02.arpa $target_dir/phones/lm.phones.filtered.02.bin
    #$KENLM_ROOT/lmplz -o 4 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback >! $target_dir/phones/lm.phones.filtered.04.arpa
    #$KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
fi

# if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
#     # python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
#     python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.02.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
# fi
