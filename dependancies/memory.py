import datetime

def cull_old_memory(messages_array, max_age_hours=4):
    if len(messages_array) <= 1:
        return messages_array
    current_time = datetime.datetime.now()
    for i in range(len(messages_array) - 1, 0, -1):
        msg = messages_array[i]
        if 'content' in msg and msg['content'].startswith('['):
            try:
                time_str = msg['content'].split(']')[0][1:] 
                msg_time = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
                msg_datetime = datetime.datetime.combine(current_time.date(), msg_time)
                if msg_datetime > current_time:
                    msg_datetime -= datetime.timedelta(days=1)
                age = (current_time - msg_datetime).total_seconds() / 3600
                if age > max_age_hours:
                    messages_array.pop(i)
            except ValueError:
                continue
    return messages_array
