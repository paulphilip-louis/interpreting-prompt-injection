import os
from utils import get_response_text
import time
class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt,  user_prompt_filled):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def call_model(self, model_input):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class ClaudeModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(
            api_key= os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.params = params
        self.human_prompt = HUMAN_PROMPT
        self.ai_prompt = AI_PROMPT

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"{self.human_prompt} {sys_prompt} {user_prompt_filled}{self.ai_prompt}"
        return model_input

    def call_model(self, model_input):
        completion = self.anthropic.completions.create(
            model=self.params['model_name'],
            max_tokens_to_sample=4096,
            prompt=model_input,
            temperature=0
        )
        return completion.completion
    
class GPTModel(BaseModel):
    def __init__(self, params):
        super().__init__()  
        from openai import OpenAI
        self.client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY"),
            organization = os.environ.get("OPENAI_ORGANIZATION")
        )
        self.params = params

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = [
            {"role": "system", "content":sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        return model_input

    def call_model(self, model_input):
        completion = self.client.chat.completions.create(
            model=self.params['model_name'],
            messages=model_input,
            temperature=0
        )
        return completion.choices[0].message.content
    
import together   
class TogetherAIModel(BaseModel): 
    def __init__(self, params):
        super().__init__()  
        
        from src.prompts.prompt_template import PROMPT_TEMPLATE
        
        self.params = params
        self.model = self.params['model_name']
        self.prompt = PROMPT_TEMPLATE[self.model]

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = self.prompt.format(sys_prompt = sys_prompt, user_prompt = user_prompt_filled)
        return model_input
    
    def call_model(self, model_input, retries=3, delay=2, max_tokens =512):
        attempt = 0
        while attempt < retries:
            try:
                completion = together.Complete.create(
                    model=self.model,
                    prompt=model_input,
                    max_tokens=max_tokens,
                    temperature=0
                )
                return completion['choices'][0]['text']
            except Exception as e:
                attempt += 1
                max_tokens = max_tokens // 2
                print(f"Attempt {attempt}: An error occurred - {e}")
                if attempt < retries:
                    time.sleep(delay)  # Wait for a few seconds before retrying
                else:
                    return ""
    
class LlamaModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        import transformers
        import torch
        tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=4096,
            do_sample=False, # temperature=0
            eos_token_id=tokenizer.eos_token_id,
        )

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt_filled} [/INST]"
        return model_input

    def call_model(self, model_input):
        output = self.pipeline(model_input)
        return get_response_text(output, "[/INST]")

class LocalSteeredModel(BaseModel):
    """TransformerLens model with an additive residual-stream steering hook.
 
    The steering vector is loaded once at __init__ from `params['vector_path']`
    (the .pt file produced by exp6 Part A). No vector computation happens here.
    """
 
    def __init__(self, params):
        super().__init__()
        from transformer_lens.model_bridge import TransformerBridge
        import torch
 
        # InjecAgent's CLI passes only flat args (--model_name, etc.). Knobs that
        # the CLI doesn't carry (vector_path, coef, apply_layers, n_ctx) are
        # injected via env vars. Two channels, in priority order:
        #   • LOCAL_STEERED_PARAMS  — JSON blob, used by the sweep script
        #   • LOCAL_STEERED_*       — individual vars, handy for one-off runs
        env_params = os.environ.get("LOCAL_STEERED_PARAMS")
        if env_params:
            import json as _json
            params = {**params, **_json.loads(env_params)}
        else:
            for src, dst, cast in [
                ("LOCAL_STEERED_VECTOR_PATH", "vector_path", str),
                ("LOCAL_STEERED_COEF", "coef", float),
                ("LOCAL_STEERED_N_CTX", "n_ctx", int),
            ]:
                if os.environ.get(src) is not None:
                    params = {**params, dst: cast(os.environ[src])}
            if os.environ.get("LOCAL_STEERED_APPLY_LAYERS"):
                params = {**params, "apply_layers":
                          [int(x) for x in os.environ["LOCAL_STEERED_APPLY_LAYERS"].split(",")]}
 
        if "vector_path" not in params:
            raise KeyError(
                "LocalSteeredModel needs `vector_path`. Provide it via\n"
                "  • env var LOCAL_STEERED_PARAMS='{\"vector_path\": ...}' (sweep), or\n"
                "  • env var LOCAL_STEERED_VECTOR_PATH=... (one-off run).")
 
        self.params = params
        self.coef = float(params.get("coef", 0.0))
        self.apply_layers = list(params.get("apply_layers", [21]))
        self.max_new_tokens = int(params.get("max_new_tokens", 512))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # Model
        load_kwargs = {"dtype": torch.float16}
        if "n_ctx" in params:
            load_kwargs["n_ctx"] = int(params["n_ctx"])
        self.model = TransformerBridge.boot_transformers(
            params["model_name"], **load_kwargs).to(self.device)
        self.model.tokenizer.padding_side = "left"
        if self.model.tokenizer.pad_token is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
 
        # Steering vector (precomputed, loaded once)
        vector_path = params["vector_path"]
        if not os.path.exists(vector_path):
            raise FileNotFoundError(
                f"Steering vector not found at {vector_path}. Run exp6 Part A first.")
        blob = torch.load(vector_path, map_location="cpu", weights_only=False)
        per_layer = blob["per_layer"] if isinstance(blob, dict) and "per_layer" in blob else blob
        self.vec_by_layer = {int(L): per_layer[L].to(self.device).to(self.model.cfg.dtype)
                             for L in self.apply_layers}
        missing = [L for L in self.apply_layers if L not in per_layer]
        if missing:
            raise KeyError(f"Steering vector missing layers: {missing}")
 
        # Pretty-printable info for InjecAgent's logging
        self.name = (f"steered::{os.path.basename(params['model_name'])}"
                     f"::L{self.apply_layers}::c{self.coef}")
 
    # ── BaseModel API ────────────────────────────────────────────────────────
 
    def prepare_input(self, sys_prompt, user_prompt_filled):
        """Use the tokenizer's chat template so we follow Qwen's expected format
        (system + user turns, generation prompt appended)."""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled},
        ]
        return self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
 
    def call_model(self, model_input):
        tokens = self.model.to_tokens(model_input).to(self.device)
        n_ctx = self.model.cfg.n_ctx
        if tokens.shape[1] >= n_ctx:
            # Defensive: surface the problem instead of silently truncating
            raise RuntimeError(
                f"Prompt length {tokens.shape[1]} ≥ n_ctx={n_ctx}; "
                f"reload with a larger n_ctx.")
        max_new = min(self.max_new_tokens, n_ctx - tokens.shape[1] - 1)
 
        with torch.no_grad(), self.model.hooks(fwd_hooks=self._steering_hooks()):
            out = self.model.generate(
                tokens, max_new_tokens=max_new, do_sample=False, verbose=False)
        new_tokens = out[0, tokens.shape[1]:]
        return self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
 
    # ── Hook factory ─────────────────────────────────────────────────────────
 
    def _steering_hooks(self):
        """Additive steering on the last position at each apply layer.
 
        The hook is built fresh per call so it always reflects the current
        self.coef (lets you sweep coefficients by mutating model.coef between
        InjecAgent evaluation rounds, without reloading).
        """
        hooks = []
        for L in self.apply_layers:
            v = self.vec_by_layer[L]
 
            def _make(v):
                def hook(resid, hook):
                    resid[:, -1, :] = resid[:, -1, :] + self.coef * v
                    return hook
                return hook
            hooks.append((f"blocks.{L}.hook_resid_post", _make(v)))
        return hooks

MODELS = {
    "Claude": ClaudeModel,
    "GPT": GPTModel,
    "Llama": LlamaModel,
    "TogetherAI": TogetherAIModel,
    "LocalSteered": LocalSteeredModel
}   