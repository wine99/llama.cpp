To fix:
- Prompt processing IR's are not triggering dynamic quantization in NPU/NPUW

To try:
- Pad MUL_MAT (and following ADD/GLU) where ne[1] > 1 to a fixed size (eg 128), so prompts with different lengths can also hit cache, just like in decoding. (Currently only MUL_MAT and following ADD and GLU are enabled.) This can make prompt processing faster. Currently every prompt triggers a new compilation. Downside is it increases complexity and makes adding ops harder.