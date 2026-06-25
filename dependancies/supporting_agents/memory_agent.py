import ollama
from dependancies.nlp import calculate_semantic_similarity
from dependancies.memory import cull_old_memory

def consolidate_memory(messages, agent_reply_original, COMPARISON_BRAIN, SUMMARY_BRAIN, SUMMARY_MODEL_MAX_TOKENS):
    summary_triggered = False
    last_agent_message = None
    last_agent_index = -1

    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'assistant':
            last_agent_message = messages[i]['content']
            last_agent_index = i
            break

    similarity_score = calculate_semantic_similarity(agent_reply_original, last_agent_message, COMPARISON_BRAIN)
    print(f"\033[94m[Score] Similarity: {similarity_score:.2f}\033[0m")

    comparison_payload = None
    agent_reply_final = agent_reply_original

    if similarity_score > 0.8:
        print(f"\033[96m[System] High semantic overlap detected ({similarity_score:.2f}). Triggering memory consolidation...\033[0m")
        
        combined_text = f"Action 1: {last_agent_message}\nAction 2: {agent_reply_original}\n\n[SYSTEM DIRECTIVE: Merge these two observations into a single, brief sentence. Do not hallucinate or add outside details]"
        
        try:
            summary_response = ollama.generate(
                model=SUMMARY_BRAIN, 
                prompt=combined_text,
                options={
                    'num_predict': SUMMARY_MODEL_MAX_TOKENS,
                    'num_thread': 4,
                    'num_gpu': 99
                }
            )

            summary_triggered = True
            consolidated_reply = summary_response['response'].strip()
            print(f"\033[95m[Summary Agent]: {consolidated_reply}\033[0m")
            
            comparison_payload = {
                "last_msg": last_agent_message,
                "current_msg": agent_reply_original,
                "score": f"{similarity_score:.2f}",
                "summary_msg": consolidated_reply
            }

            prev_user_index = last_agent_index - 1
            if prev_user_index > 0 and messages[prev_user_index].get('role') == 'user':
                curr_user_msg = messages[-1]
                prev_user_msg = messages[prev_user_index]
                if curr_user_msg.get('role') == 'user':
                    curr_user_msg['content'] = f"{prev_user_msg['content']}\n\n{curr_user_msg['content']}"
                    if 'images' in prev_user_msg:
                        if 'images' not in curr_user_msg:
                            curr_user_msg['images'] = []
                        curr_user_msg['images'] = prev_user_msg['images'] + curr_user_msg['images']
                messages.pop(last_agent_index)
                messages.pop(prev_user_index)
            else:
                messages.pop(last_agent_index)
                 
            agent_reply_final = consolidated_reply
            
        except Exception as e:
            print(f"\033[91m[System] Summarization Error: {e}. Falling back to default deduplication.\033[0m")
            prev_user_index = last_agent_index - 1
            if prev_user_index > 0 and messages[prev_user_index].get('role') == 'user':
                curr_user_msg = messages[-1]
                prev_user_msg = messages[prev_user_index]
                if curr_user_msg.get('role') == 'user':
                    curr_user_msg['content'] = f"{prev_user_msg['content']}\n\n{curr_user_msg['content']}"
                    if 'images' in prev_user_msg:
                        if 'images' not in curr_user_msg:
                            curr_user_msg['images'] = []
                        curr_user_msg['images'] = prev_user_msg['images'] + curr_user_msg['images']
                messages.pop(last_agent_index)
                messages.pop(prev_user_index)
            else:
                messages.pop(last_agent_index)

    if summary_triggered == False:
        print(f"\033[97m[Agent]: {agent_reply_final}\033[0m")

    messages.append({
        'role': 'assistant',
        'content': agent_reply_final
    })

    return comparison_payload, messages

def enforce_memory_limits(messages, max_age_hours=4.0, max_items=40):
    messages = cull_old_memory(messages, max_age_hours=max_age_hours)
    if len(messages) > max_items: 
        messages.pop(1)
        messages.pop(1)
    return messages
