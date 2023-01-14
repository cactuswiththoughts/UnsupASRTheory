#!/bin/bash

gan_type=$1
graph_type=$2
gen_type=$3  # {"unigram_avg", "cnn"}
discrim_type=$4  # {"mlp", "cnn", "linear"}
stage=$5
stop_stage=$6

if [ -z $gen_type ]; then
    gen_type="unigram_avg"
fi
if [ -z $discrim_type ]; then
    discrim_type="mlp"
fi
if [ $gan_type = l1 ]; then 
    discrim_type=perfect
fi
if [ -z $stage ]; then
    stage=0
    stop_stage=100
fi

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
gpu_num=0
export root
export checkpoint_root
export FAIRSEQ_ROOT
export KALDI_ROOT
export KENLM_ROOT

graph_name=${graph_type}
if [ $graph_name = "hypercube" ]; then
    nx=8
else
    nx=10
fi
graph_type=${graph_name}__nx_${nx}_n_2
level=3
bsz=2560

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    n=51200
    #in_path=$root/manifest/phase_transition_${graph_name}/${graph_type}_Nx_${n}_level_${level} 
    #zsh scripts/prepare_synthetic.sh $in_path 1 \
    #|| error "prepare_synthetic.sh failed for n=${n}, l=${l}"
    for n_speech in 51200 12800 3200 800 200; do
        #for n_text in 51200 12800 3200 800 200; do
            n_text=${n_speech}
            raw_path=$root/manifest/phase_transition_${graph_name}/${graph_type}_Nx_${n}_level_${level} 
            in_path=$root/manifest/phase_transition_${graph_name}/${graph_type}_Nx_${n_speech}_Ny_${n_text}_level_${level}
            python scripts/subset_synthetic.py \
                --in_path ${raw_path} \
                --speech_subset_size ${n_speech} \
                --text_subset_size ${n_text} \
                --out_path ${in_path}
            zsh scripts/prepare_synthetic.sh $in_path 1 \
            || error "prepare_synthetic.sh failed for n_speech=${n_speech}, n_text=${n_text}"
        #done
    done
fi

# The effect of resetting discriminator
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    PREFIX=w2v_unsup_gan_xp
    for l in 0 1 2 3 4 5 6 7 8 9; do    
        tgt_dir=manifest/phase_transition_${graph_name}/${graph_type}_Nx_2560_level_${l}
        checkpoint_dir=${checkpoint_root}/${graph_type}_Nx_2560_level_${l}_gan_type_${gan_type}_bsz${bsz}_wo_reset
        TASK_DATA=$root/$tgt_dir
        TEXT_DATA=$root/$tgt_dir/phones  # path to fairseq-preprocessed GAN data (phones dir)
        KENLM_PATH=$root/$tgt_dir/phones/lm.phones.filtered.02.bin #kenlm.phn.o4.bin  # KenLM bi-gram phoneme language model (LM data = GAN data here)
        if [ $graph_name = "debruijn" ]; then
            config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_l80_wo_reset
        elif [ $graph_name = "hypercube" ]; then
            config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_nx8_wo_reset
        else
            config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_wo_reset
        fi
        echo ${config_name}

        CUDA_VISIBLE_DEVICES=${gpu_num} PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            -m --config-dir config/gan \
            --config-name ${config_name} \
            task.data=${TASK_DATA} \
            task.text_data=${TEXT_DATA} \
            task.kenlm_path=${KENLM_PATH} \
            common.user_dir=$(pwd)/unsupervised \
            model.code_penalty=0.0 model.gradient_penalty=0.0 \
            model.smoothness_weight=0.0 'common.seed=range(0,1)' \
            checkpoint.save_dir=${checkpoint_dir} \
            hydra.run.dir=${checkpoint_dir} \
            hydra.sweep.dir=${checkpoint_dir}
        rm -r pymp*
    done
fi

# The effect of sampling without replacement
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    PREFIX=w2v_unsup_gan_xp
    l=${level}
    tgt_dir=manifest/phase_transition_${graph_name}/${graph_type}_Nx_2560_level_${l}
    checkpoint_dir=${checkpoint_root}/${graph_type}_Nx_2560_level_${l}_gan_type_${gan_type}_bsz${bsz}_random_choice
    TASK_DATA=$root/$tgt_dir
    TEXT_DATA=$root/$tgt_dir/phones  # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=$root/$tgt_dir/phones/lm.phones.filtered.02.bin #kenlm.phn.o4.bin  # KenLM bi-gram phoneme language model (LM data = GAN data here)
    if [ $graph_name = "debruijn" ]; then
        config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_l80_random_choice
    elif [ $graph_name = "hypercube" ]; then
        config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_nx8_random_choice
    else
        config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_random_choice
    fi
    echo ${config_name}

    CUDA_VISIBLE_DEVICES=${gpu_num} PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/gan \
        --config-name ${config_name} \
        task.data=${TASK_DATA} \
        task.text_data=${TEXT_DATA} \
        task.kenlm_path=${KENLM_PATH} \
        common.user_dir=$(pwd)/unsupervised \
        model.code_penalty=0.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=0.0 'common.seed=range(0,1)' \
        checkpoint.save_dir=${checkpoint_dir} \
        hydra.run.dir=${checkpoint_dir} \
        hydra.sweep.dir=${checkpoint_dir}
fi

# The effect of discriminator hidden width
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    PREFIX=w2v_unsup_gan_xp
    l=${level}
    tgt_dir=manifest/phase_transition_${graph_name}/${graph_type}_Nx_2560_level_${l}
    checkpoint_dir=${checkpoint_root}/${graph_type}_Nx_2560_level_${l}_gan_type_${gan_type}_bsz${bsz}_2layer_discrim
    TASK_DATA=$root/$tgt_dir
    TEXT_DATA=$root/$tgt_dir/phones  # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=$root/$tgt_dir/phones/lm.phones.filtered.02.bin #kenlm.phn.o4.bin  # KenLM bi-gram phoneme language model (LM data = GAN data here)
    if [ $graph_name = "debruijn" ]; then
        config_name=w2vu_synthetic_${gan_type}_gan_switch_freq2_bsz${bsz}_l80
    elif [ $graph_name = "hypercube" ]; then
        config_name=w2vu_synthetic_${gan_type}_gan_switch_freq2_bsz${bsz}_nx8
    else
        config_name=w2vu_synthetic_${gan_type}_gan_switch_freq2_bsz${bsz}
    fi
    echo ${config_name}

    CUDA_VISIBLE_DEVICES=${gpu_num} PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/gan \
        --config-name ${config_name} \
        task.data=${TASK_DATA} \
        task.text_data=${TEXT_DATA} \
        task.kenlm_path=${KENLM_PATH} \
        common.user_dir=$(pwd)/unsupervised \
        model.code_penalty=0.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=0.0 'common.seed=range(0,1)' \
        checkpoint.save_dir=${checkpoint_dir} \
        hydra.run.dir=${checkpoint_dir} \
        hydra.sweep.dir=${checkpoint_dir}
fi

# The effect of sample size
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    PREFIX=w2v_unsup_gan_xp
    l=${level}
    for n_speech in 51200 12800 3200 800 200; do
        for n_text in 51200 12800 3200 800 200; do
            tgt_dir=manifest/phase_transition_${graph_name}/${graph_type}_Nx_${n_speech}_Ny_${n_text}_level_${l}
            checkpoint_dir=${checkpoint_root}/${graph_type}_Nx_${n_speech}_Ny_${n_text}_level_${l}_gan_type_${gan_type}_bsz${bsz}
            TASK_DATA=$root/$tgt_dir
            TEXT_DATA=$root/$tgt_dir/phones  # path to fairseq-preprocessed GAN data (phones dir)
            KENLM_PATH=$root/$tgt_dir/phones/lm.phones.filtered.02.bin #kenlm.phn.o4.bin  # KenLM bi-gram phoneme language model (LM data = GAN data here)
            if [ $graph_name = "debruijn" ]; then
                config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_l80_random_choice
            elif [ $graph_name = "hypercube" ]; then
                config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_nx8_random_choice
            else
                config_name=w2vu_synthetic_${gan_type}_gan_linear_discrim_switch_freq2_bsz${bsz}_random_choice
            fi
            echo ${config_name}

            CUDA_VISIBLE_DEVICES=${gpu_num} PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
                -m --config-dir config/gan \
                --config-name ${config_name} \
                task.data=${TASK_DATA} \
                task.text_data=${TEXT_DATA} \
                task.kenlm_path=${KENLM_PATH} \
                common.user_dir=$(pwd)/unsupervised \
                model.code_penalty=0.0 model.gradient_penalty=0.0 \
                model.smoothness_weight=0.0 'common.seed=range(0,1)' \
                checkpoint.save_dir=${checkpoint_dir} \
                hydra.run.dir=${checkpoint_dir} \
                hydra.sweep.dir=${checkpoint_dir}
        done
    done
fi

# The effect of generator type
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
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
            config_name=w2vu_synthetic_${gan_type}_${gen_type}_gan_${discrim_type}_discrim_switch_freq2_bsz2560_l80
        elif [ $graph_name = "hypercube" ]; then
            config_name=w2vu_synthetic_${gan_type}_${gen_type}_gan_${discrim_type}_discrim_switch_freq2_bsz2560_nx8
        else
            config_name=w2vu_synthetic_${gan_type}_${gen_type}_gan_${discrim_type}_discrim_switch_freq2_bsz2560
        fi

        # config_name=w2vu_synthetic_perfect_d_l1_bsz${bsz}
        CUDA_VISIBLE_DEVICES=${gpu_num} PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            -m --config-dir config/gan \
            --config-name ${config_name} \
            task.data=${TASK_DATA} \
            task.text_data=${TEXT_DATA} \
            task.kenlm_path=${KENLM_PATH} \
            common.user_dir=$(pwd)/unsupervised \
            model.code_penalty=0.0 model.gradient_penalty=0.0 \
            model.smoothness_weight=0.0 'common.seed=range(0,1)' \
            checkpoint.save_dir=${checkpoint_dir} \
            hydra.run.dir=${checkpoint_dir} \
            hydra.sweep.dir=${checkpoint_dir}
    done
fi
