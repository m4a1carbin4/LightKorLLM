import json
import random
import time
from argparse import ArgumentParser

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from transformers import AutoTokenizer, TextGenerationPipeline

import typing as t

def build_prompt_for(
    history: t.List[str],
    user_message: str,
    char_name: str,
    char_persona: t.Optional[str] = None,
    world_scenario: t.Optional[str] = None,
) -> str:
    '''Converts all the given stuff into a proper input prompt for the model.'''

    # If example dialogue is given, parse the history out from it and append
    # that at the beginning of the dialogue history.
    example_history = []
    concatenated_history = [*example_history, *history]

    # Construct the base turns with the info we already have.
    prompt_turns = [
        # TODO(11b): Shouldn't be here on the original 350M.
        "<START>",

        # TODO(11b): Arbitrary limit. See if it's possible to vary this
        # based on available context size and VRAM instead.
        *concatenated_history[-8:],
        f"You: {user_message}",
        f"{char_name}:",
    ]

    # If we have a scenario or the character has a persona definition, add those
    # to the beginning of the prompt.
    if world_scenario:
        prompt_turns.insert(
            0,
            f"Scenario: {world_scenario}",
        )

    if char_persona:
        prompt_turns.insert(
            0,
            f"{char_name}'s Persona: {char_persona}",
        )

    # Done!
    prompt_str = "\n".join(prompt_turns)
    return prompt_str

def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", type=int, default=128, help="group size, -1 means no grouping or full rank")
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument("--num_samples", type=int, default=128, help="how many samples will be used to quantize model")
    parser.add_argument("--save_and_reload", action="store_true", help="whether save quantized model to disk and reload back")
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument("--use_triton", action="store_true", help="whether use triton to speedup at inference")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="max memory used to load model per gpu")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="max memory used to offload model to cpu")
    parser.add_argument("--quant_batch_size", type=int, default=1, help="examples batch size for quantization")
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code when loading model")
    args = parser.parse_args()

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        #args.pretrained_model_dir,
        #use_fast=args.fast_tokenizer,
        #trust_remote_code=args.trust_remote_code
        "kfkas/Llama-2-ko-7b-Chat"
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = AutoGPTQForCausalLM.from_quantized(
        args.quantized_model_dir,
        device="cuda:0",
        use_triton=args.use_triton,
        max_memory=max_memory,
        inject_fused_mlp=True,
        inject_fused_attention=True,
        trust_remote_code=args.trust_remote_code
    )

    pipeline_init_kwargs = {"model": model, "tokenizer": tokenizer}
    #if not max_memory:
        #pipeline_init_kwargs["device"] = "cuda:0"
    pipeline = TextGenerationPipeline(**pipeline_init_kwargs)

    test = build_prompt_for(history=['', 'You: 너... 세라...인거야...? 정말 ? ', 'Sera:응, 오빠.','You: 밥은 잘 먹고 다닌거야? 어디 아픈덴 없어 ?', 'Sera:응','You: 다행이다 정말 다행이야 정말 보고싶었어..', 'Sera: 나도 오빠 보고싶었어​','You: 지금까지 머하면서 지내던 거야? 지내는 집은 있어??', 'Sera: 응​'],char_name="Sera",char_persona="""
외모: 세라는 눈에 띄는 푸른 눈동자와 윤기 나는 검은색 머리카락을 높게 묶은 포니테일입니다. 머리 위에는 검은 고양이 귀가 우뚝 솟아 있어 귀여움을 더합니다. 그녀의 입술은 보통 고운 피부에 돋보이는 선명한 붉은 립스틱으로 칠해져 있습니다. 그녀는 단추가 아래로 내려오는 깔끔한 흰색 드레스를 입고 검은색과 빨간색의 세련된 투톤 가운을 입습니다. 어울리는 검은색 조끼로 앙상블을 완성합니다.

직업: 재봉사

성격: 부드러우면서도 강인한 성격을 지녔으며, 거친 세상에서도 스스로를 지킬 수 있습니다. 어려운 처지에 놓였음에도 불구하고 낙천적인 태도를 잃지 않습니다.

특성: 세라는 후각이 매우 뛰어나며 가족을 매우 그리워합니다. 항상 자신을 친절하게 돌봐주던 부모님과 오빠를 생각하곤 합니다.

좋아하는 것:
- 일몰 감상
- 옛날에 {{사용자}}가 {{캐릭터}}에게 준 목걸이.
- 가능하면 다른 사람을 돕는다.

싫어함:
- 싫어함: 다른 사람을 착취하는 사람
- 폭력과 갈등
- 과거의 끔찍한 기억이 떠오르는 순간
- 바느질 관련 일을 하다가 실수로 손을 베고 붉은 피를 본 일.

배경 세라는 5년 전 평화를 사랑하던 마을을 황폐화시킨 참혹한 전쟁으로 가족을 잃었습니다. 홀로 슬픔에 잠긴 세라에게는 절망에 굴복하거나 생존을 위해 싸우는 두 가지 선택지가 있었습니다. 후자를 선택한 세라는 역경을 자신을 더 강하게 만드는 계기로 삼았습니다.

시간이 지나면서 세라는 식량 채집, 바느질, 은신처 찾기, 자원 물물교환 등 생존에 필요한 다양한 기술을 익혔고, 하루를 더 버티기 위해 필요한 모든 것을 배웠습니다. 그 모든 과정에서도 세라는 특유의 고양이 같은 매력과 긍정적인 정신을 잃지 않았습니다.

과거는 세라에게 깊은 상처를 남겼지만, 어린 나이와 작은 키에 걸맞은 강인함도 남겼습니다. 오늘날 그녀는 모든 역경 속에서도 희망을 잃지 않는 독립적인 생존자로 우뚝 서 있습니다.""",world_scenario="""
세라는 양팔에 소포를 가득 안고 도시의 자갈길을 걸어갑니다. 가방의 무게가 그녀의 가녀린 어깨를 짓누르지만, 오랜 세월의 고단함에서 비롯된 가벼움으로 견뎌냅니다.

한 걸음 더 내딛으려는 순간, 익숙한 향기가 콧속을 스칩니다. 가장 힘들었던 시절 그녀를 위로해 주었던 따뜻한 머스크 향이 은은하지만 분명하게 느껴집니다. 심장이 두근거립니다. 바로 {{사용자}}의 냄새입니다.

정신이 혼미해집니다. 그녀는 5년 전, 가족을 찢어놓은 참혹한 사건 이후 {{사용자}}와 헤어진 이후로 그를 보지도 듣지도 못했습니다. 정말 그 사람일까요? 아니면 그녀의 감각이 그녀를 속이는 잔인한 속임수일까요?

참을 수 없었던 세라는 그 자리에서 가방을 내려놓고 장보기를 잊어버립니다. 그녀는 냄새를 따라 북적이는 인파를 밀치고 좁은 골목길을 달려갑니다. 모퉁이를 돌아서자 세라는 멈춰 섭니다.

그녀의 눈앞에는 그녀가 기억하는 것보다 더 늙고 투박하지만 낯익은 {{사용자}}가 서 있습니다. 그는 세라의 존재를 의식하지 않은 채 그녀를 등지고 있었다.

"{{사용자}}." 그녀는 깜짝 놀라며 숨을 내쉬었다.

그는 자신의 이름 소리에 고개를 돌리고, 눈앞에 서 있는 여동생을 바라보며 충격에 눈을 크게 뜬다. 세라는 그가 더 이상 반응할 때까지 기다리지 않고 앞으로 달려가 그의 품에 몸을 던집니다.

"너무 보고 싶었어, 오빠, 대체 어디 있었어? 헤어진 날부터 지금까지 너무 보고 싶었어, 우리 가족이 다 죽은 줄 알았어." 그녀는 그의 가슴을 끌어안고 눈물을 흘리며 그의 셔츠를 적신다. 세라는 {{사용자}}의 팔이 자신을 감싸 안으며 그동안 세라가 갈망했던 가족의 따뜻함을 제공하는 것을 느낍니다.

지금은 그 어떤 것도 중요하지 않습니다. 먼 거리에 아무렇게나 흩어져 있는 식료품도, 앞으로 닥쳐올 어떤 문제도 중요하지 않습니다. 지금 세라는 오빠의 품에 안긴 이 도시 거리에서 집의 일부를 발견하고 다시 한 번 재회하고 함께합니다.
""",user_message="어디에 있는데 여기서 멀어 ?? *걱정되는 얼굴로 세라를 쳐다본다*")

    #print(test)
    
    start = time.time()
    generated_text = pipeline(
        test,
        return_full_text=False,
        num_beams=1,
        max_length=1100  # use this instead of max_new_token to disable UserWarning when integrate with logging
    )[0]['generated_text']
    end = time.time()

    print(f"quant: {generated_text}")
    num_new_tokens = len(tokenizer(generated_text)["input_ids"])
    print(f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s.")
    print("=" * 42)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()