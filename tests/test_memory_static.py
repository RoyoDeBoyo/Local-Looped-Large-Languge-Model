import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dependancies.memory import cull_old_memory

def main():
    print("\033[96m=== Memory Culling Test ===\033[0m")
    
    max_age_input = input("\033[96mEnter max age in hours (default 4.0): \033[0m").strip()
    max_age_hours = float(max_age_input) if max_age_input else 4.0
    
    num_items_input = input("\033[96mEnter number of synthetic memory items (default 50): \033[0m").strip()
    num_items = int(num_items_input) if num_items_input else 50
    
    print(f"\n\033[93mGenerating {num_items} synthetic memory items spanning the last 10 hours...\033[0m")
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    now = datetime.datetime.now()
    
    # Generate items from 10 hours ago up to now
    for i in range(num_items):
        hours_ago = 10.0 - (10.0 * (i / max(1, num_items - 1)))
        item_time = now - datetime.timedelta(hours=hours_ago)
        time_str = item_time.strftime("%H:%M:%S")
        
        messages.append({
            "role": "user",
            "content": f"[{time_str}] [User Question]: This is synthetic memory item {i} created {hours_ago:.1f} hours ago."
        })
        messages.append({
            "role": "assistant",
            "content": f"This is the agent response for item {i}."
        })

    print(f"\033[92mGenerated {len(messages)} total messages (including system prompt).\033[0m")
    print(f"Oldest message is from approx 10 hours ago.")
    print(f"Newest message is from approx 0 hours ago.")
    
    print(f"\n\033[96mRunning cull_old_memory(max_age_hours={max_age_hours})...\033[0m")
    culled_messages = cull_old_memory(messages.copy(), max_age_hours=max_age_hours)
    
    print(f"\033[92mAfter culling: {len(culled_messages)} total messages remain.\033[0m")
    
    if len(culled_messages) > 1:
        # Check the oldest user message
        oldest_user_msg = next((msg for msg in culled_messages if msg['role'] == 'user'), None)
        if oldest_user_msg:
            print(f"\033[90mOldest remaining user message: {oldest_user_msg['content']}\033[0m")
            
    print("\n\033[92mMemory Culling Test Complete.\033[0m")

if __name__ == "__main__":
    main()
