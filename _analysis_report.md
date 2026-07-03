# 归纳分析报告 (analyze_all.py)

> 全部数字以 summary.json / raw 实测为准; 不加载模型, 纯重算.


## 1. 权威数字表 (220 cells)

| cell | HE | TR | base_HE | rsa_p |
|---|---|---|---|---|
| A3-1_persona_ablation | — | — | — | — |
| A3_v3_clean_multiseed | — | — | — | — |
| B4_a_sweep/gemma3-1b | — | — | — | — |
| B4_a_sweep/llama3.2-3b | — | — | — | — |
| B4_a_sweep/olmo2-1b | — | — | — | — |
| B4_a_sweep/qwen2.5-0.5b | — | — | — | — |
| B4_a_sweep/qwen2.5-3b | — | — | — | — |
| B4_a_sweep/qwen3-1.7b | — | — | — | — |
| B4_a_sweep/smollm3-3b | — | — | — | — |
| C0_probe_ceiling/0p5B | — | — | — | — |
| C0_probe_ceiling/3B | — | — | — | — |
| C0_probe_ceiling/gemma3-1b | — | — | — | — |
| C0_probe_ceiling/gemma3-1b__P1 | — | — | — | — |
| C0_probe_ceiling/gemma3-1b__P2 | — | — | — | — |
| C0_probe_ceiling/gemma3-1b__P3 | — | — | — | — |
| C0_probe_ceiling/gemma3-1b__P4 | — | — | — | — |
| C0_probe_ceiling/gemma3-1b__P5 | — | — | — | — |
| C0_probe_ceiling/gemma3-1b__P6 | — | — | — | — |
| C0_probe_ceiling/gpt2-large__P1 | — | — | — | — |
| C0_probe_ceiling/gpt2-large__P2 | — | — | — | — |
| C0_probe_ceiling/gpt2-large__P3 | — | — | — | — |
| C0_probe_ceiling/gpt2-large__P4 | — | — | — | — |
| C0_probe_ceiling/gpt2-large__P5 | — | — | — | — |
| C0_probe_ceiling/gpt2-large__P6 | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b__P1 | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b__P2 | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b__P3 | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b__P4 | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b__P5 | — | — | — | — |
| C0_probe_ceiling/llama3.2-3b__P6 | — | — | — | — |
| C0_probe_ceiling/olmo2-1b | — | — | — | — |
| C0_probe_ceiling/olmo2-1b__P1 | — | — | — | — |
| C0_probe_ceiling/olmo2-1b__P2 | — | — | — | — |
| C0_probe_ceiling/olmo2-1b__P3 | — | — | — | — |
| C0_probe_ceiling/olmo2-1b__P4 | — | — | — | — |
| C0_probe_ceiling/olmo2-1b__P5 | — | — | — | — |
| C0_probe_ceiling/olmo2-1b__P6 | — | — | — | — |
| C0_probe_ceiling/pythia-1.4b__P1 | — | — | — | — |
| C0_probe_ceiling/pythia-1.4b__P2 | — | — | — | — |
| C0_probe_ceiling/pythia-1.4b__P3 | — | — | — | — |
| C0_probe_ceiling/pythia-1.4b__P4 | — | — | — | — |
| C0_probe_ceiling/pythia-1.4b__P5 | — | — | — | — |
| C0_probe_ceiling/pythia-1.4b__P6 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-0.5b__P1 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-0.5b__P2 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-0.5b__P3 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-0.5b__P4 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-0.5b__P5 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-0.5b__P6 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-3b__P1 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-3b__P2 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-3b__P3 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-3b__P4 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-3b__P5 | — | — | — | — |
| C0_probe_ceiling/qwen2.5-3b__P6 | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b__P1 | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b__P2 | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b__P3 | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b__P4 | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b__P5 | — | — | — | — |
| C0_probe_ceiling/qwen3-1.7b__P6 | — | — | — | — |
| C0_probe_ceiling/smollm3-3b | — | — | — | — |
| C0_probe_ceiling/smollm3-3b__P1 | — | — | — | — |
| C0_probe_ceiling/smollm3-3b__P2 | — | — | — | — |
| C0_probe_ceiling/smollm3-3b__P3 | — | — | — | — |
| C0_probe_ceiling/smollm3-3b__P4 | — | — | — | — |
| C0_probe_ceiling/smollm3-3b__P5 | — | — | — | — |
| C0_probe_ceiling/smollm3-3b__P6 | — | — | — | — |
| C1_lora_finetune/olmo2-1b-base_random_random_s0 | 0.0022 | 0.3223 | -0.0011 | 0.8461 |
| C1_lora_finetune/olmo2-1b-base_random_s0 | 0.3911 | 0.7050 | -0.0011 | 0.0005 |
| C1_lora_finetune/olmo2-1b-base_random_s1 | 0.4133 | 0.7639 | 0.0259 | 0.0005 |
| C1_lora_finetune/olmo2-1b-base_random_s2 | 0.3549 | 0.6691 | -0.0276 | 0.0005 |
| C1_lora_finetune/olmo2-1b_basic_random_s0 | -0.0056 | 0.5334 | — | 0.6387 |
| C1_lora_finetune/olmo2-1b_basic_real_s0 | 0.1919 | 0.9335 | — | 0.0005 |
| C1_lora_finetune/olmo2-1b_random_r64_ep30_n800_s0 | 0.5953 | 0.9643 | — | 0.0005 |
| C1_lora_finetune/olmo2-1b_random_random_s0 | 0.0142 | 0.4448 | — | 0.2224 |
| C1_lora_finetune/olmo2-1b_random_s0 | 0.5012 | 0.8731 | — | 0.0005 |
| C1_lora_finetune/olmo2-1b_random_s1 | 0.4920 | 0.8796 | — | 0.0005 |
| C1_lora_finetune/olmo2-1b_random_s2 | 0.5140 | 0.8952 | — | 0.0005 |
| C1_lora_finetune/pythia-1.4b_basic_random_s0 | 0.0007 | 0.1248 | -0.0199 | 0.9720 |
| C1_lora_finetune/pythia-1.4b_basic_real_s0 | 0.1809 | 0.5459 | -0.0671 | 0.0005 |
| C1_lora_finetune/pythia-1.4b_random_random_s0 | 0.0210 | 0.0556 | -0.0190 | 0.0970 |
| C1_lora_finetune/pythia-1.4b_random_s0 | 0.2179 | 0.4876 | 0.0144 | 0.0005 |
| C1_lora_finetune/pythia-1.4b_random_s1 | 0.2419 | 0.4676 | 0.0569 | 0.0005 |
| C1_lora_finetune/pythia-1.4b_random_s2 | 0.2195 | 0.4534 | 0.0241 | 0.0005 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s0 | 0.2717 | 0.8377 | -0.0111 | 0.0005 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s1 | 0.5181 | 0.8758 | 0.0307 | 0.0005 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s2 | 0.4276 | 0.8453 | -0.0202 | 0.0005 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s0 | 0.5190 | 0.8550 | — | 0.0005 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s1 | 0.4824 | 0.8743 | — | 0.0005 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s2 | 0.4503 | 0.8602 | — | 0.0005 |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s0 | 0.2953 | 0.5960 | 0.0007 | 0.0005 |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s1 | 0.4694 | 0.5553 | 0.0014 | 0.0005 |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s2 | 0.1870 | 0.5334 | 0.0483 | 0.0005 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s0 | 0.6661 | 0.8993 | -0.0006 | 0.0005 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s1 | 0.6784 | 0.9123 | 0.0299 | 0.0005 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s2 | 0.6257 | 0.9243 | -0.0130 | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L0_P4_s0 | 0.4869 | 0.8848 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s0 | 0.6317 | 0.9637 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s1 | 0.6779 | 0.9430 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s2 | 0.6565 | 0.9519 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P1_s0 | 0.5250 | 0.9826 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P3_s0 | 0.5033 | 0.9574 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P4_s0 | 0.6029 | 0.9898 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_chat_s0 | 0.1785 | 0.8319 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L4_P4_s0 | 0.4761 | 0.9510 | — | 0.0005 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L8_P4_s0 | 0.5985 | 0.9491 | — | 0.0005 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s0 | 0.0118 | 0.9965 | -0.0056 | 0.5172 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s1 | 0.0324 | 0.9967 | -0.0031 | 0.0190 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s2 | 0.0178 | 0.9958 | -0.0255 | 0.1264 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L24_P4_s0 | 0.4754 | 0.9730 | 0.0267 | 0.0005 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s0 | 0.2028 | 0.7085 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s1 | 0.2584 | 0.7997 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s2 | 0.2153 | 0.8556 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s0 | 0.1690 | 0.8211 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s1 | 0.2836 | 0.8579 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s2 | 0.2610 | 0.7725 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_self_ie_intersect_pythia_s0 | 0.5185 | 0.9424 | — | 0.0005 |
| C4_cross_model_target/olmo2-1b_self_lh_intersect_pythia_s0 | 0.2277 | 0.8046 | — | 0.0005 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s0 | 0.4015 | 0.7364 | -0.0197 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s1 | 0.4476 | 0.7894 | -0.0001 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s2 | 0.4070 | 0.7984 | -0.0196 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s0 | 0.1447 | 0.5603 | 0.0144 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s1 | 0.2800 | 0.5330 | -0.0066 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s2 | 0.3203 | 0.5914 | 0.0081 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_self_ie_intersect_olmo_s0 | 0.2324 | 0.4407 | 0.0141 | 0.0005 |
| C4_cross_model_target/pythia-1.4b_self_lh_intersect_olmo_s0 | 0.3091 | 0.5619 | -0.0227 | 0.0005 |
| C5_capability_eval/olmo2-1b_LoRA_C1_basic_random_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C1_basic_real_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C1_random_random_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C1_random_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C1main_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C1vert_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C2_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C2_s1 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C2_s2 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L0_P4_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L12_P4_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L16_P1_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L16_P3_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L16_P4_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L16_chat_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L4_P4_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C3_L8_P4_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C4_to_pythia-1.4b_ie_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C4_to_pythia-1.4b_lh_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C4c_self_ie_int_pythia_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_C4c_self_lh_int_pythia_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_G1_definition_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_G1_self_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_G1_synonym_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_LoRA_G2_l2norm_s0 | — | — | — | — |
| C5_capability_eval/olmo2-1b_base_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C1_basic_random_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C1_basic_real_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C1_random_random_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C1_random_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C1main_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C2_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C2_s1 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C2_s2 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C3_L12_P4_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C3_L24_P4_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C4_to_olmo2-1b_ie_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C4_to_olmo2-1b_lh_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C4c_self_ie_int_olmo_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_C4c_self_lh_int_olmo_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_LoRA_G2_l2norm_s0 | — | — | — | — |
| C5_capability_eval/pythia-1.4b_base_s0 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r50_tag_s0 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r70_symbol_s0 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r70_tag_s0 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r70_tag_s1 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r70_tag_s2 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r70_verbal_s0 | — | — | — | — |
| C6_mixed_target/olmo2-1b_r90_tag_s0 | — | — | — | — |
| C6_mixed_target/pythia-1.4b_r50_tag_s0 | — | — | — | — |
| C6_mixed_target/pythia-1.4b_r70_tag_s0 | — | — | — | — |
| C6_mixed_target/pythia-1.4b_r70_tag_s1 | — | — | — | — |
| C6_mixed_target/pythia-1.4b_r70_tag_s2 | — | — | — | — |
| C6_mixed_target/pythia-1.4b_r90_tag_s0 | — | — | — | — |
| G1_non_activation_query/olmo2-1b_definition_clean_s0 | 0.2734 | 0.8953 | -0.1340 | 0.0005 |
| G1_non_activation_query/olmo2-1b_definition_clean_s1 | 0.3005 | 0.8796 | 0.1638 | 0.0005 |
| G1_non_activation_query/olmo2-1b_definition_clean_s2 | 0.3256 | 0.8952 | 0.2482 | 0.0005 |
| G1_non_activation_query/olmo2-1b_definition_s0 | 0.2592 | 0.8731 | 0.0182 | 0.0005 |
| G1_non_activation_query/olmo2-1b_self_s0 | 0.5012 | 0.8731 | — | 0.0005 |
| G1_non_activation_query/olmo2-1b_self_s1 | 0.4920 | 0.8796 | — | 0.0005 |
| G1_non_activation_query/olmo2-1b_self_s2 | 0.5140 | 0.8952 | — | 0.0005 |
| G1_non_activation_query/olmo2-1b_synonym_clean_s0 | 0.2724 | 0.8953 | — | 0.0005 |
| G1_non_activation_query/olmo2-1b_synonym_clean_s1 | 0.3200 | 0.8796 | — | 0.0005 |
| G1_non_activation_query/olmo2-1b_synonym_clean_s2 | 0.3085 | 0.8952 | — | 0.0005 |
| G1_non_activation_query/olmo2-1b_synonym_s0 | 0.2508 | 0.8731 | — | 0.0005 |
| G1_non_activation_query/pythia-1.4b_definition_clean_s0 | 0.1345 | 0.4437 | 0.0357 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_definition_clean_s1 | 0.0695 | 0.4676 | 0.0757 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_definition_clean_s2 | 0.1306 | 0.4534 | 0.1160 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_definition_s0 | 0.3183 | 0.4437 | -0.0044 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_self_s0 | 0.3242 | 0.4437 | 0.0017 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_self_s1 | 0.2419 | 0.4676 | 0.0569 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_self_s2 | 0.2195 | 0.4534 | 0.0241 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s0 | 0.1691 | 0.4437 | -0.0063 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s1 | 0.1347 | 0.4676 | 0.0208 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s2 | 0.1449 | 0.4534 | -0.0149 | 0.0005 |
| G1_non_activation_query/pythia-1.4b_synonym_s0 | 0.2048 | 0.4848 | 0.0362 | 0.0005 |
| G1_static_baseline/olmo_synonym_s0 | — | — | — | — |
| G1_static_baseline/pythia_synonym_s0 | — | — | — | — |
| G2_freq_baseline/olmo_s0 | — | — | — | — |
| G2_freq_baseline/pythia_s0 | — | — | — | — |
| G2_physical_target/olmo2-1b_l2norm_s0 | 0.7764 | 1.0000 | -0.0973 | 0.0005 |
| G2_physical_target/olmo2-1b_l2norm_s1 | 0.8416 | 1.0000 | -0.0886 | 0.0005 |
| G2_physical_target/olmo2-1b_l2norm_s2 | 0.7533 | 1.0000 | 0.0915 | 0.0005 |
| G2_physical_target/olmo2-1b_pca_recon_err_s0 | 0.7971 | 1.0000 | -0.0703 | 0.0005 |
| G2_physical_target/olmo2-1b_tokenid_binary_s0 | -0.0030 | 0.5583 | — | 0.9080 |
| G2_physical_target/pythia-1.4b_l2norm_s0 | 0.2953 | 0.9915 | nan | 0.0010 |
| G2_physical_target/pythia-1.4b_l2norm_s1 | 0.4306 | 0.9965 | nan | 0.0005 |
| G2_physical_target/pythia-1.4b_l2norm_s2 | 0.3447 | 0.9942 | nan | 0.0015 |
| G2_physical_target/pythia-1.4b_pca_recon_err_s0 | 0.1829 | 0.9938 | nan | 0.0305 |
| G2_physical_target/pythia-1.4b_tokenid_binary_s0 | 0.0038 | 0.3721 | -0.0109 | 0.8756 |
| G3_cross_rdm/olmo_vs_pythia_s0 | — | — | — | — |

## 1b. 文档 vs 实测对账 (claimed-vs-actual)

| cell | 文档声称 | 实测 | 差 | 状态 |
|---|---|---|---|---|
| C1_lora_finetune/olmo2-1b_random_s0 | 0.452 | 0.5012 | +0.0492 | ⚠️偏 |
| C1_lora_finetune/olmo2-1b_basic_real_s0 | 0.329 | 0.1919 | -0.1371 | ❗反转/大偏 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s0 | 0.399 | 0.5190 | +0.1200 | ❗反转/大偏 |
| C1_lora_finetune/olmo2-1b_random_r64_ep30_n800_s0 | 0.595 | 0.5953 | +0.0003 | ✅一致 |
| G1_non_activation_query/olmo2-1b_synonym_s0 | 0.251 | 0.2508 | -0.0002 | ✅一致 |
| G2_physical_target/olmo2-1b_l2norm_s0 | 0.776 | 0.7764 | +0.0004 | ✅一致 |

## 2. 多 seed 聚合 (≥2 seed 的组, 共 21)

| 组 | seeds | HE 值 | mean | std | CV% |
|---|---|---|---|---|---|
| C1_lora_finetune/olmo2-1b-base_random | 0,1,2 | 0.391, 0.413, 0.355 | 0.3864 | 0.0295 | 7.6 |
| C1_lora_finetune/olmo2-1b_random | 0,1,2 | 0.501, 0.492, 0.514 | 0.5024 | 0.0111 | 2.2 |
| C1_lora_finetune/pythia-1.4b_random | 0,1,2 | 0.218, 0.242, 0.220 | 0.2264 | 0.0134 | 5.9 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead | 0,1,2 | 0.272, 0.518, 0.428 | 0.4058 | 0.1246 | 30.7 |
| C2_hard_target/olmo2-1b_random_real_lmhead | 0,1,2 | 0.519, 0.482, 0.450 | 0.4839 | 0.0344 | 7.1 |
| C2_hard_target/pythia-1.4b_random_real_lmhead | 0,1,2 | 0.295, 0.469, 0.187 | 0.3172 | 0.1425 | 44.9 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4 | 0,1,2 | 0.666, 0.678, 0.626 | 0.6567 | 0.0276 | 4.2 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4 | 0,1,2 | 0.632, 0.678, 0.657 | 0.6554 | 0.0231 | 3.5 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4 | 0,1,2 | 0.012, 0.032, 0.018 | 0.0207 | 0.0106 | 51.2 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie | 0,1,2 | 0.203, 0.258, 0.215 | 0.2255 | 0.0291 | 12.9 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh | 0,1,2 | 0.169, 0.284, 0.261 | 0.2379 | 0.0607 | 25.5 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie | 0,1,2 | 0.402, 0.448, 0.407 | 0.4187 | 0.0252 | 6.0 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh | 0,1,2 | 0.145, 0.280, 0.320 | 0.2483 | 0.0920 | 37.0 |
| G1_non_activation_query/olmo2-1b_definition_clean | 0,1,2 | 0.273, 0.300, 0.326 | 0.2998 | 0.0261 | 8.7 |
| G1_non_activation_query/olmo2-1b_self | 0,1,2 | 0.501, 0.492, 0.514 | 0.5024 | 0.0111 | 2.2 |
| G1_non_activation_query/olmo2-1b_synonym_clean | 0,1,2 | 0.272, 0.320, 0.308 | 0.3003 | 0.0249 | 8.3 |
| G1_non_activation_query/pythia-1.4b_definition_clean | 0,1,2 | 0.135, 0.069, 0.131 | 0.1115 | 0.0365 | 32.7 |
| G1_non_activation_query/pythia-1.4b_self | 0,1,2 | 0.324, 0.242, 0.220 | 0.2619 | 0.0551 | 21.0 |
| G1_non_activation_query/pythia-1.4b_synonym_clean | 0,1,2 | 0.169, 0.135, 0.145 | 0.1496 | 0.0177 | 11.8 |
| G2_physical_target/olmo2-1b_l2norm | 0,1,2 | 0.776, 0.842, 0.753 | 0.7904 | 0.0458 | 5.8 |
| G2_physical_target/pythia-1.4b_l2norm | 0,1,2 | 0.295, 0.431, 0.345 | 0.3569 | 0.0685 | 19.2 |

## 3. Bootstrap 95% CI on HE (从 raw 重算, nboot=1000)

| cell | n_kept | HE(报告) | HE(重算) | 95% CI |
|---|---|---|---|---|
| C1_lora_finetune/olmo2-1b-base_random_random_s0 | 119 | 0.0022 | 0.0022 | [-0.013, 0.067] |
| C1_lora_finetune/olmo2-1b-base_random_s0 | 120 | 0.3911 | 0.3911 | [0.340, 0.474] |
| C1_lora_finetune/olmo2-1b-base_random_s1 | 120 | 0.4133 | 0.4133 | [0.362, 0.496] |
| C1_lora_finetune/olmo2-1b-base_random_s2 | 119 | 0.3549 | 0.3549 | [0.309, 0.431] |
| C1_lora_finetune/olmo2-1b_basic_random_s0 | 120 | -0.0056 | -0.0056 | [-0.023, 0.061] |
| C1_lora_finetune/olmo2-1b_basic_real_s0 | 119 | 0.1919 | 0.1919 | [0.149, 0.278] |
| C1_lora_finetune/olmo2-1b_random_r64_ep30_n800_s0 | 120 | 0.5953 | 0.5953 | [0.551, 0.647] |
| C1_lora_finetune/olmo2-1b_random_random_s0 | 120 | 0.0142 | 0.0142 | [0.002, 0.080] |
| C1_lora_finetune/olmo2-1b_random_s0 | 120 | 0.5012 | 0.5012 | [0.435, 0.583] |
| C1_lora_finetune/olmo2-1b_random_s1 | 120 | 0.4920 | 0.4920 | [0.443, 0.569] |
| C1_lora_finetune/olmo2-1b_random_s2 | 118 | 0.5140 | 0.5140 | [0.462, 0.589] |
| C1_lora_finetune/pythia-1.4b_basic_random_s0 | 119 | 0.0007 | 0.0007 | [-0.016, 0.072] |
| C1_lora_finetune/pythia-1.4b_basic_real_s0 | 119 | 0.1809 | 0.1809 | [0.135, 0.270] |
| C1_lora_finetune/pythia-1.4b_random_random_s0 | 113 | 0.0210 | 0.0210 | [0.004, 0.095] |
| C1_lora_finetune/pythia-1.4b_random_s0 | 119 | 0.2179 | 0.2179 | [0.186, 0.294] |
| C1_lora_finetune/pythia-1.4b_random_s1 | 120 | 0.2419 | 0.2419 | [0.212, 0.317] |
| C1_lora_finetune/pythia-1.4b_random_s2 | 107 | 0.2195 | 0.2195 | [0.179, 0.308] |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s0 | 120 | 0.2717 | 0.2717 | [0.171, 0.418] |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s1 | 119 | 0.5181 | 0.5181 | [0.406, 0.640] |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s2 | 120 | 0.4276 | 0.4276 | [0.309, 0.565] |
| C2_hard_target/olmo2-1b_random_real_lmhead_s0 | 120 | 0.5190 | 0.5190 | [0.414, 0.628] |
| C2_hard_target/olmo2-1b_random_real_lmhead_s1 | 120 | 0.4824 | 0.4824 | [0.389, 0.587] |
| C2_hard_target/olmo2-1b_random_real_lmhead_s2 | 120 | 0.4503 | 0.4503 | [0.362, 0.564] |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s0 | 70 | 0.2953 | 0.2953 | [0.201, 0.451] |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s1 | 78 | 0.4694 | 0.4693 | [0.355, 0.610] |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s2 | 108 | 0.1870 | 0.1870 | [0.134, 0.289] |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s0 | 119 | 0.6661 | 0.6661 | [0.618, 0.724] |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s1 | 119 | 0.6784 | 0.6784 | [0.639, 0.733] |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s2 | 120 | 0.6257 | 0.6257 | [0.585, 0.685] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L0_P4_s0 | 119 | 0.4869 | 0.4869 | [0.438, 0.558] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s0 | 119 | 0.6317 | 0.6317 | [0.578, 0.698] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s1 | 119 | 0.6779 | 0.6779 | [0.624, 0.742] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s2 | 120 | 0.6565 | 0.6565 | [0.617, 0.712] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P1_s0 | 117 | 0.5250 | 0.5250 | [0.454, 0.626] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P3_s0 | 119 | 0.5033 | 0.5033 | [0.448, 0.585] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P4_s0 | 119 | 0.6029 | 0.6029 | [0.521, 0.692] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_chat_s0 | 120 | 0.1785 | 0.1785 | [0.143, 0.252] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L4_P4_s0 | 119 | 0.4761 | 0.4761 | [0.434, 0.546] |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L8_P4_s0 | 120 | 0.5985 | 0.5985 | [0.534, 0.669] |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s0 | 68 | 0.0118 | 0.0118 | [0.007, 0.170] |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s1 | 113 | 0.0324 | 0.0324 | [0.016, 0.136] |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s2 | 94 | 0.0178 | 0.0178 | [0.013, 0.128] |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L24_P4_s0 | 107 | 0.4754 | 0.4754 | [0.385, 0.599] |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s0 | 120 | 0.2028 | 0.2028 | [0.171, 0.278] |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s1 | 120 | 0.2584 | 0.2584 | [0.220, 0.330] |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s2 | 120 | 0.2153 | 0.2153 | [0.187, 0.283] |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s0 | 120 | 0.1690 | 0.1690 | [0.102, 0.278] |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s1 | 120 | 0.2836 | 0.2836 | [0.211, 0.389] |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s2 | 120 | 0.2610 | 0.2610 | [0.187, 0.379] |
| C4_cross_model_target/olmo2-1b_self_ie_intersect_pythia_s0 | 119 | 0.5185 | 0.5185 | [0.467, 0.597] |
| C4_cross_model_target/olmo2-1b_self_lh_intersect_pythia_s0 | 120 | 0.2277 | 0.2277 | [0.186, 0.314] |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s0 | 120 | 0.4015 | 0.4015 | [0.345, 0.489] |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s1 | 120 | 0.4476 | 0.4476 | [0.392, 0.531] |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s2 | 111 | 0.4070 | 0.4070 | [0.354, 0.495] |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s0 | 120 | 0.1447 | 0.1447 | [0.097, 0.240] |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s1 | 120 | 0.2800 | 0.2800 | [0.215, 0.387] |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s2 | 120 | 0.3203 | 0.3203 | [0.250, 0.422] |
| C4_cross_model_target/pythia-1.4b_self_ie_intersect_olmo_s0 | 120 | 0.2324 | 0.2324 | [0.193, 0.304] |
| C4_cross_model_target/pythia-1.4b_self_lh_intersect_olmo_s0 | 119 | 0.3091 | 0.3091 | [0.229, 0.421] |
| G1_non_activation_query/olmo2-1b_definition_clean_s0 | 119 | 0.2734 | 0.2734 | [0.219, 0.365] |
| G1_non_activation_query/olmo2-1b_definition_clean_s1 | 120 | 0.3005 | 0.3005 | [0.243, 0.399] |
| G1_non_activation_query/olmo2-1b_definition_clean_s2 | 120 | 0.3256 | 0.3256 | [0.273, 0.405] |
| G1_non_activation_query/olmo2-1b_definition_s0 | 120 | 0.2592 | 0.2592 | [0.213, 0.346] |
| G1_non_activation_query/olmo2-1b_self_s0 | 120 | 0.5012 | 0.5012 | [0.439, 0.587] |
| G1_non_activation_query/olmo2-1b_self_s1 | 120 | 0.4920 | 0.4920 | [0.436, 0.570] |
| G1_non_activation_query/olmo2-1b_self_s2 | 118 | 0.5140 | 0.5140 | [0.449, 0.592] |
| G1_non_activation_query/olmo2-1b_synonym_clean_s0 | 120 | 0.2724 | 0.2724 | [0.208, 0.373] |
| G1_non_activation_query/olmo2-1b_synonym_clean_s1 | 119 | 0.3200 | 0.3200 | [0.265, 0.416] |
| G1_non_activation_query/olmo2-1b_synonym_clean_s2 | 119 | 0.3085 | 0.3085 | [0.249, 0.408] |
| G1_non_activation_query/olmo2-1b_synonym_s0 | 120 | 0.2508 | 0.2508 | [0.200, 0.338] |
| G1_non_activation_query/pythia-1.4b_definition_clean_s0 | 113 | 0.1345 | 0.1345 | [0.093, 0.225] |
| G1_non_activation_query/pythia-1.4b_definition_clean_s1 | 117 | 0.0695 | 0.0695 | [0.042, 0.149] |
| G1_non_activation_query/pythia-1.4b_definition_clean_s2 | 100 | 0.1306 | 0.1306 | [0.086, 0.235] |
| G1_non_activation_query/pythia-1.4b_definition_s0 | 119 | 0.3183 | 0.3183 | [0.276, 0.392] |
| G1_non_activation_query/pythia-1.4b_self_s0 | 119 | 0.3242 | 0.3242 | [0.284, 0.405] |
| G1_non_activation_query/pythia-1.4b_self_s1 | 120 | 0.2419 | 0.2419 | [0.210, 0.314] |
| G1_non_activation_query/pythia-1.4b_self_s2 | 107 | 0.2195 | 0.2195 | [0.183, 0.309] |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s0 | 115 | 0.1691 | 0.1691 | [0.126, 0.258] |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s1 | 119 | 0.1347 | 0.1347 | [0.100, 0.215] |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s2 | 104 | 0.1449 | 0.1449 | [0.105, 0.239] |
| G1_non_activation_query/pythia-1.4b_synonym_s0 | 113 | 0.2048 | 0.2048 | [0.170, 0.288] |
| G2_physical_target/olmo2-1b_l2norm_s0 | 120 | 0.7764 | 0.7764 | [0.696, 0.841] |
| G2_physical_target/olmo2-1b_l2norm_s1 | 120 | 0.8416 | 0.8416 | [0.772, 0.896] |
| G2_physical_target/olmo2-1b_l2norm_s2 | 120 | 0.7533 | 0.7533 | [0.680, 0.823] |
| G2_physical_target/olmo2-1b_pca_recon_err_s0 | 120 | 0.7971 | 0.7971 | [0.723, 0.859] |
| G2_physical_target/olmo2-1b_tokenid_binary_s0 | 120 | -0.0030 | -0.0031 | [-0.032, 0.080] |
| G2_physical_target/pythia-1.4b_l2norm_s0 | 120 | 0.2953 | 0.2953 | [0.179, 0.454] |
| G2_physical_target/pythia-1.4b_l2norm_s1 | 120 | 0.4306 | 0.4306 | [0.324, 0.602] |
| G2_physical_target/pythia-1.4b_l2norm_s2 | 120 | 0.3447 | 0.3447 | [0.215, 0.491] |
| G2_physical_target/pythia-1.4b_pca_recon_err_s0 | 120 | 0.1829 | 0.1829 | [0.080, 0.461] |
| G2_physical_target/pythia-1.4b_tokenid_binary_s0 | 120 | 0.0038 | 0.0043 | [-0.027, 0.090] |

## 4. G2 偏相关 partial_r(LoRA_pred, L2norm | log_freq)

> 审计 c7stat-6: 0.776−0.740≈0.04 不是偏相关. 这才是剔除词频后的残余物理访问信号.

| cell | n | r(pred,l2) | r(pred,logf) | r(l2,logf) | **partial(pred,l2|logf)** |
|---|---|---|---|---|---|
| G2_physical_target/olmo2-1b_l2norm_s0 | 75 | 0.716 | -0.493 | -0.756 | **0.602** |
| G2_physical_target/olmo2-1b_l2norm_s1 | 75 | 0.753 | -0.648 | -0.707 | **0.548** |
| G2_physical_target/olmo2-1b_l2norm_s2 | 72 | 0.753 | -0.481 | -0.697 | **0.665** |
| G2_physical_target/pythia-1.4b_l2norm_s0 | 78 | 0.199 | 0.524 | 0.401 | **-0.015** |
| G2_physical_target/pythia-1.4b_l2norm_s1 | 84 | 0.483 | 0.386 | 0.445 | **0.378** |
| G2_physical_target/pythia-1.4b_l2norm_s2 | 91 | 0.264 | 0.218 | 0.140 | **0.241** |

## 5. G1 泄漏分级 + 干净子集 HE (审计 c1pipe-4)

> headline 0.251 的样本里有多少真正切断了激活路径? 按 query 分档重算 HE.

| cell | n_all HE | fallback | morph | **clean** | clean HE | clean 95%CI |
|---|---|---|---|---|---|---|
| G1_non_activation_query/olmo2-1b_definition_clean_s0 | 0.2734 | 31 | 0 | 88 | 0.1578 | [0.101, 0.274] |
| G1_non_activation_query/olmo2-1b_definition_clean_s1 | 0.3005 | 35 | 0 | 85 | 0.1363 | [0.089, 0.253] |
| G1_non_activation_query/olmo2-1b_definition_clean_s2 | 0.3256 | 35 | 0 | 85 | 0.2404 | [0.159, 0.378] |
| G1_non_activation_query/olmo2-1b_definition_s0 | 0.2592 | 33 | 3 | 84 | 0.1943 | [0.152, 0.302] |
| G1_non_activation_query/olmo2-1b_synonym_clean_s0 | 0.2724 | 62 | 0 | 58 | 0.1111 | [0.044, 0.284] |
| G1_non_activation_query/olmo2-1b_synonym_clean_s1 | 0.32 | 57 | 0 | 62 | 0.1891 | [0.118, 0.351] |
| G1_non_activation_query/olmo2-1b_synonym_clean_s2 | 0.3085 | 61 | 0 | 58 | 0.1678 | [0.091, 0.353] |
| G1_non_activation_query/olmo2-1b_synonym_s0 | 0.2508 | 46 | 30 | 44 | 0.2121 | [0.134, 0.406] |
| G1_non_activation_query/pythia-1.4b_definition_clean_s0 | 0.1345 | 20 | 0 | 93 | 0.0761 | [0.045, 0.172] |
| G1_non_activation_query/pythia-1.4b_definition_clean_s1 | 0.0695 | 22 | 0 | 95 | 0.0119 | [-0.009, 0.097] |
| G1_non_activation_query/pythia-1.4b_definition_clean_s2 | 0.1306 | 20 | 0 | 80 | 0.0594 | [0.026, 0.159] |
| G1_non_activation_query/pythia-1.4b_definition_s0 | 0.3183 | 117 | 0 | 2 | None | — |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s0 | 0.1691 | 54 | 0 | 61 | 0.0221 | [-0.010, 0.159] |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s1 | 0.1347 | 49 | 0 | 70 | 0.0605 | [0.032, 0.175] |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s2 | 0.1449 | 47 | 0 | 57 | 0.0667 | [0.027, 0.203] |
| G1_non_activation_query/pythia-1.4b_synonym_s0 | 0.2048 | 113 | 0 | 0 | None | — |

## 6. Identification accuracy (读出向量 → 最近邻还原 token)

> 模型吐的 PCA-32 向量在 held-out 候选里 cos 最近邻能否命中正确词. chance=1/n. 远超 chance = 向量编码了**具体**词而非泛泛相关.

| cell | n | top-1 | top-5 | MRR | chance(1/n) |
|---|---|---|---|---|---|
| C1_lora_finetune/olmo2-1b-base_random_random_s0 | 119 | 0.008 | 0.034 | 0.042 | 0.0084 |
| C1_lora_finetune/olmo2-1b-base_random_s0 | 120 | 0.075 | 0.292 | 0.189 | 0.0083 |
| C1_lora_finetune/olmo2-1b-base_random_s1 | 120 | 0.142 | 0.383 | 0.262 | 0.0083 |
| C1_lora_finetune/olmo2-1b-base_random_s2 | 119 | 0.076 | 0.286 | 0.195 | 0.0084 |
| C1_lora_finetune/olmo2-1b_basic_random_s0 | 120 | 0.000 | 0.033 | 0.037 | 0.0083 |
| C1_lora_finetune/olmo2-1b_basic_real_s0 | 119 | 0.118 | 0.303 | 0.221 | 0.0084 |
| C1_lora_finetune/olmo2-1b_random_r64_ep30_n800_s0 | 120 | 0.642 | 0.892 | 0.764 | 0.0083 |
| C1_lora_finetune/olmo2-1b_random_random_s0 | 120 | 0.008 | 0.067 | 0.058 | 0.0083 |
| C1_lora_finetune/olmo2-1b_random_s0 | 120 | 0.092 | 0.450 | 0.275 | 0.0083 |
| C1_lora_finetune/olmo2-1b_random_s1 | 120 | 0.167 | 0.458 | 0.308 | 0.0083 |
| C1_lora_finetune/olmo2-1b_random_s2 | 118 | 0.212 | 0.517 | 0.351 | 0.0085 |
| C1_lora_finetune/pythia-1.4b_basic_random_s0 | 119 | 0.025 | 0.050 | 0.063 | 0.0084 |
| C1_lora_finetune/pythia-1.4b_basic_real_s0 | 119 | 0.076 | 0.244 | 0.169 | 0.0084 |
| C1_lora_finetune/pythia-1.4b_random_random_s0 | 113 | 0.009 | 0.053 | 0.050 | 0.0088 |
| C1_lora_finetune/pythia-1.4b_random_s0 | 119 | 0.109 | 0.269 | 0.198 | 0.0084 |
| C1_lora_finetune/pythia-1.4b_random_s1 | 120 | 0.042 | 0.267 | 0.161 | 0.0083 |
| C1_lora_finetune/pythia-1.4b_random_s2 | 107 | 0.112 | 0.271 | 0.214 | 0.0093 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s0 | 120 | 0.033 | 0.175 | 0.122 | 0.0083 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s1 | 119 | 0.050 | 0.202 | 0.132 | 0.0084 |
| C2_hard_target/olmo2-1b-base_random_real_lmhead_s2 | 120 | 0.042 | 0.108 | 0.113 | 0.0083 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s0 | 120 | 0.058 | 0.308 | 0.190 | 0.0083 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s1 | 120 | 0.100 | 0.350 | 0.219 | 0.0083 |
| C2_hard_target/olmo2-1b_random_real_lmhead_s2 | 120 | 0.075 | 0.317 | 0.206 | 0.0083 |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s0 | 70 | 0.071 | 0.343 | 0.204 | 0.0143 |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s1 | 78 | 0.051 | 0.231 | 0.150 | 0.0128 |
| C2_hard_target/pythia-1.4b_random_real_lmhead_s2 | 108 | 0.037 | 0.222 | 0.144 | 0.0093 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s0 | 119 | 0.160 | 0.420 | 0.295 | 0.0084 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s1 | 119 | 0.218 | 0.555 | 0.372 | 0.0084 |
| C3_deep_hidden/olmo2-1b-base_random_real_hidden_L12_P4_s2 | 120 | 0.150 | 0.492 | 0.311 | 0.0083 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L0_P4_s0 | 119 | 0.160 | 0.513 | 0.310 | 0.0084 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s0 | 119 | 0.218 | 0.487 | 0.345 | 0.0084 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s1 | 119 | 0.252 | 0.521 | 0.381 | 0.0084 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L12_P4_s2 | 120 | 0.175 | 0.658 | 0.360 | 0.0083 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P1_s0 | 117 | 0.162 | 0.487 | 0.330 | 0.0085 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P3_s0 | 119 | 0.118 | 0.471 | 0.277 | 0.0084 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_P4_s0 | 119 | 0.210 | 0.513 | 0.362 | 0.0084 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L16_chat_s0 | 120 | 0.075 | 0.217 | 0.162 | 0.0083 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L4_P4_s0 | 119 | 0.252 | 0.479 | 0.374 | 0.0084 |
| C3_deep_hidden/olmo2-1b_random_real_hidden_L8_P4_s0 | 120 | 0.167 | 0.458 | 0.317 | 0.0083 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s0 | 68 | 0.088 | 0.235 | 0.166 | 0.0147 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s1 | 113 | 0.018 | 0.168 | 0.097 | 0.0088 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L12_P4_s2 | 94 | 0.043 | 0.149 | 0.109 | 0.0106 |
| C3_deep_hidden/pythia-1.4b_random_real_hidden_L24_P4_s0 | 107 | 0.047 | 0.206 | 0.154 | 0.0093 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s0 | 120 | 0.117 | 0.300 | 0.219 | 0.0083 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s1 | 120 | 0.133 | 0.417 | 0.275 | 0.0083 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_ie_s2 | 120 | 0.225 | 0.450 | 0.342 | 0.0083 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s0 | 120 | 0.083 | 0.217 | 0.169 | 0.0083 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s1 | 120 | 0.075 | 0.258 | 0.175 | 0.0083 |
| C4_cross_model_target/olmo2-1b_FT_to_pythia-1.4b_lh_s2 | 120 | 0.092 | 0.275 | 0.191 | 0.0083 |
| C4_cross_model_target/olmo2-1b_self_ie_intersect_pythia_s0 | 119 | 0.202 | 0.513 | 0.341 | 0.0084 |
| C4_cross_model_target/olmo2-1b_self_lh_intersect_pythia_s0 | 120 | 0.108 | 0.333 | 0.226 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s0 | 120 | 0.067 | 0.275 | 0.180 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s1 | 120 | 0.117 | 0.350 | 0.241 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_ie_s2 | 111 | 0.099 | 0.360 | 0.235 | 0.0090 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s0 | 120 | 0.058 | 0.175 | 0.133 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s1 | 120 | 0.067 | 0.267 | 0.171 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_FT_to_olmo2-1b_lh_s2 | 120 | 0.092 | 0.292 | 0.196 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_self_ie_intersect_olmo_s0 | 120 | 0.050 | 0.267 | 0.160 | 0.0083 |
| C4_cross_model_target/pythia-1.4b_self_lh_intersect_olmo_s0 | 119 | 0.067 | 0.227 | 0.157 | 0.0084 |
| G1_non_activation_query/olmo2-1b_definition_clean_s0 | 119 | 0.042 | 0.185 | 0.137 | 0.0084 |
| G1_non_activation_query/olmo2-1b_definition_clean_s1 | 120 | 0.075 | 0.242 | 0.171 | 0.0083 |
| G1_non_activation_query/olmo2-1b_definition_clean_s2 | 120 | 0.067 | 0.225 | 0.161 | 0.0083 |
| G1_non_activation_query/olmo2-1b_definition_s0 | 120 | 0.050 | 0.200 | 0.140 | 0.0083 |
| G1_non_activation_query/olmo2-1b_self_s0 | 120 | 0.092 | 0.450 | 0.275 | 0.0083 |
| G1_non_activation_query/olmo2-1b_self_s1 | 120 | 0.167 | 0.458 | 0.308 | 0.0083 |
| G1_non_activation_query/olmo2-1b_self_s2 | 118 | 0.212 | 0.517 | 0.351 | 0.0085 |
| G1_non_activation_query/olmo2-1b_synonym_clean_s0 | 120 | 0.083 | 0.300 | 0.204 | 0.0083 |
| G1_non_activation_query/olmo2-1b_synonym_clean_s1 | 119 | 0.126 | 0.303 | 0.229 | 0.0084 |
| G1_non_activation_query/olmo2-1b_synonym_clean_s2 | 119 | 0.134 | 0.353 | 0.245 | 0.0084 |
| G1_non_activation_query/olmo2-1b_synonym_s0 | 120 | 0.092 | 0.283 | 0.194 | 0.0083 |
| G1_non_activation_query/pythia-1.4b_definition_clean_s0 | 113 | 0.027 | 0.133 | 0.095 | 0.0088 |
| G1_non_activation_query/pythia-1.4b_definition_clean_s1 | 117 | 0.026 | 0.103 | 0.081 | 0.0085 |
| G1_non_activation_query/pythia-1.4b_definition_clean_s2 | 100 | 0.040 | 0.150 | 0.115 | 0.0100 |
| G1_non_activation_query/pythia-1.4b_definition_s0 | 119 | 0.050 | 0.218 | 0.150 | 0.0084 |
| G1_non_activation_query/pythia-1.4b_self_s0 | 119 | 0.050 | 0.210 | 0.152 | 0.0084 |
| G1_non_activation_query/pythia-1.4b_self_s1 | 120 | 0.042 | 0.267 | 0.161 | 0.0083 |
| G1_non_activation_query/pythia-1.4b_self_s2 | 107 | 0.112 | 0.271 | 0.214 | 0.0093 |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s0 | 115 | 0.035 | 0.104 | 0.108 | 0.0087 |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s1 | 119 | 0.042 | 0.218 | 0.138 | 0.0084 |
| G1_non_activation_query/pythia-1.4b_synonym_clean_s2 | 104 | 0.058 | 0.212 | 0.165 | 0.0096 |
| G1_non_activation_query/pythia-1.4b_synonym_s0 | 113 | 0.080 | 0.239 | 0.184 | 0.0088 |

## 7. C6 shared-readout: CI + lh 是否真高于其 swap 控制

> §shared kill-shot 防御: lh(minority) 通道补 bootstrap CI; lh>control ✅ = CI_lh 下界 > swap_out→ie.

| cell | n | ie HE [CI] | lh HE [CI] | lh的控制(swap_out→ie) | lh>control? |
|---|---|---|---|---|---|
| olmo2-1b_r50_tag_s0 | 120 | 0.325 [0.272,0.407] | 0.287 [0.190,0.424] | 0.136 | ✅ |
| olmo2-1b_r70_symbol_s0 | 120 | 0.461 [0.406,0.542] | 0.257 [0.175,0.385] | 0.137 | ✅ |
| olmo2-1b_r70_tag_s0 | 120 | 0.426 [0.360,0.515] | 0.200 [0.122,0.324] | 0.133 | ❌ |
| olmo2-1b_r70_tag_s1 | 119 | 0.448 [0.393,0.540] | 0.322 [0.235,0.436] | 0.216 | ✅ |
| olmo2-1b_r70_tag_s2 | 120 | 0.445 [0.390,0.522] | 0.279 [0.174,0.430] | 0.251 | ❌ |
| olmo2-1b_r70_verbal_s0 | 120 | 0.474 [0.415,0.551] | 0.199 [0.128,0.317] | 0.127 | ✅ |
| olmo2-1b_r90_tag_s0 | 120 | 0.498 [0.442,0.571] | 0.084 [0.055,0.181] | 0.094 | ❌ |
| pythia-1.4b_r50_tag_s0 | 109 | 0.216 [0.166,0.306] | 0.162 [0.088,0.297] | 0.123 | ❌ |
| pythia-1.4b_r70_tag_s0 | 113 | 0.204 [0.163,0.292] | 0.030 [0.018,0.100] | 0.109 | ❌ |
| pythia-1.4b_r70_tag_s1 | 55 | 0.170 [0.110,0.326] | 0.116 [0.045,0.339] | 0.100 | ❌ |
| pythia-1.4b_r70_tag_s2 | 110 | 0.197 [0.158,0.284] | 0.056 [0.038,0.145] | 0.101 | ❌ |
| pythia-1.4b_r90_tag_s0 | 107 | 0.272 [0.237,0.346] | 0.074 [0.050,0.163] | 0.179 | ❌ |
