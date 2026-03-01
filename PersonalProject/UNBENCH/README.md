# UNBench Reproducibility Project

FTEC5660 Reproducibility Work — [UNBench: Benchmarking LLMs for Political Science](https://arxiv.org/pdf/2502.14122) (AAAI 2026)

## Setup

```bash
cd UNBENCH
pip install -r requirements.txt
```

## API Keys

| Provider | Env Var | Model |
|----------|---------|-------|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini, gpt-4o |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| Qwen | `DASHSCOPE_API_KEY` | qwen-plus, qwen-turbo |

## Quick Start (Reproducibility)

1. **Set API key** (at least OpenAI for baseline):
   ```bash
   export OPENAI_API_KEY=your_key
   export DEEPSEEK_API_KEY=your_key
   export DASHSCOPE_API_KEY=your_key
   ```

2. **Download full data** (optional, for ~30 samples):
   ```bash
   bash download_data.sh
   ```

3. **Run full experiment** (OpenAI baseline + DeepSeek + Qwen + parameter modifications):
   ```bash
   python run_full_experiment.py
   ```

4. **Results** in `results/experiments.json`.

## Individual Runs

**OpenAI (baseline):**
```bash
python run_task3.py --provider openai --model gpt-4o-mini --temperature 0
```

**DeepSeek:**
```bash
python run_task3.py --provider deepseek --model deepseek-chat --temperature 0
```

**Qwen (通义千问, DashScope):**
```bash
export DASHSCOPE_API_KEY=your_key
python run_task3.py --provider qwen --model qwen-plus --temperature 0
```

**Parameter modifications:**
```bash
# Temperature 0 → 0.7 (3 trials)
python run_task3.py --provider openai --model gpt-4o-mini --temperature 0.7 --trials 3

# max_tokens 5 → 50
python run_task3.py --provider openai --model gpt-4o-mini --max-tokens 50
```

## Paper-Aligned Features

Per [UNBench paper](https://arxiv.org/pdf/2502.14122):

1. **Multi-model comparison** — GPT-4o-mini, DeepSeek, Qwen (Table 2: GPT-4o Bal.ACC 0.677, Mac.F1 0.363)
2. **Metadata ablation** — Add `--metadata` when data has date/authors
3. **Multiple trials** — Use `--trials 3` for temp>0 to report mean±std
4. **Parameter modifications** — Temperature, max_tokens

