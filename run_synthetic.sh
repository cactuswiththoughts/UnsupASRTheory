#!/bin/bash

gan_type=$1
graph_name=$2
gen_type=$3
discrim_type=$4
if [ -z $gen_type ]; then
    gen_type="avg"
fi
if [ -z $discrim_type ]; then
    discrim_type=linear
fi
if [ $gan_type = l1 ]; then 
    discrim_type=perfect
fi

#level=$3
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

CONDA_ROOT=  # =====> The location of your Anaconda 
FAIRSEQ_ROOT=  # =====> The location of your Fairseq
KALDI_ROOT=  # =====> The location of your Kaldi
KENLM_ROOT=  # =====> The the location of your KenLM 

source ${CONDA_ROOT}/anaconda3/etc/profile.d/conda.sh
conda activate fairseq
root=$(pwd)
checkpoint_root=${root}/multirun
#graph_type=debruijn_4_4 
#graph_type=directed_cycle_10
if [ $graph_name = "hypercube" ]; then
    nx=8
else
    nx=10
fi
graph_type=${graph_name}__nx_${nx}_n_2  #disjoint_cycles__nx_10_n_2

gpu_num=3
export root
export checkpoint_root
export FAIRSEQ_ROOT
export KALDI_ROOT
export KENLM_ROOT

stage=0
stop_stage=100
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for level in 0; do
        for n in 2560; do
            in_path=$root/manifest/phase_transition_${graph_name}/${graph_type}_Nx_${n}_level_${level} 
            zsh scripts/prepare_synthetic.sh $in_path 1 \
            || error "prepare_synthetic.sh failed for n=${n}, l=${l}"
        done
    done
fi

# GAN training with an imperfect discriminator
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    PREFIX=w2v_unsup_gan_xp 
    #l=${level}
    for l in 0 1 2 3 4 5 6 7 8 9; do
        tgt_dir=manifest/phase_transition_${graph_name}/${graph_type}_Nx_2560_level_${l}
        checkpoint_dir=${checkpoint_root}/${graph_type}_Nx_2560_level_${l}_gan_type_${gen_type}_${discrim_type}_${gan_type}
        TASK_DATA=$root/$tgt_dir
        echo ${TASK_DATA}
        TEXT_DATA=$root/$tgt_dir/phones  # path to fairseq-preprocessed GAN data (phones dir)
        KENLM_PATH=$root/$tgt_dir/phones/lm.phones.filtered.02.bin #kenlm.phn.o4.bin  # KenLM bi-gram phoneme language model (LM data = GAN data here)
        if [ $graph_name = "debruijn" ]; then
            config_name=w2vu_synthetic_${gan_type}_gan_${discrim_type}_discrim_switch_freq2_bsz2560_l80
        elif [ $graph_name = "hypercube" ]; then
            config_name=w2vu_synthetic_${gan_type}_gan_${discrim_type}_discrim_switch_freq2_bsz2560_nx8
        else
            config_name=w2vu_synthetic_${gan_type}_gan_${discrim_type}_discrim_switch_freq2_bsz2560
        fi
        echo $config_name

        # config_name=w2vu_synthetic_perfect_d_l1_bsz${bsz}
        if [ -d $checkpoint_dir/1 ]; then
            rm -r $checkpoint_dir/1
        fi
        if [ -d $checkpoint_dir/2 ]; then
            rm -r $checkpoint_dir/2
        fi

        CUDA_VISIBLE_DEVICES=${gpu_num} PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            -m --config-dir config/gan \
            --config-name ${config_name} \
            task.data=${TASK_DATA} \
            task.text_data=${TEXT_DATA} \
            task.kenlm_path=${KENLM_PATH} \
            common.user_dir=$(pwd)/unsupervised \
            model.code_penalty=0.0 model.gradient_penalty=0.0 \
            model.smoothness_weight=0.0 'common.seed=range(0,3)' \
            hydra.run.dir=${checkpoint_dir} \
            hydra.sweep.dir=${checkpoint_dir}
#            checkpoint.save_dir=${checkpoint_dir} \
    done
fi
