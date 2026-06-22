# LLLLM Architecture

## Disclaimer
**This model has been based on how I as a human percieve the world around me. This document as been altered by AI to make it understandable and segmented into different sections.**

## Introduction
When I started with this project, I wanted to be able to interact with an AI model on my own hardware. It was an extension of the Cloudflare bot that I'd been working on, and I wanted to be able to use it myself. Tools like Alexa and Siri have had the ability to interact with real-world appliances for years; however, they haven't been able to comprehend and understand the world around them to actually fit a person's needs.

## The Biological Approach to AI Senses
In order to do this, I started with drafting how I interact with the world. Each part of your body can be considered as a separate "model." The eyes that you use to see can be considered as a dedicated "vision model," your ears a dedicated "hearing model," your mouth as a dedicated "speaking model," your tongue as a dedicated "tasting model," and so on and so forth. These are all linked together to your brain, which is the "master model." Even the brain is split up into different sections that do different things. While this line of thinking is not backed up by research, this is simply how I understood it.

I started thinking about how I would actually implement this into an AI model and AI architecture and came up with this idea:
You have several models in the same "group chat" that have their own roles and send their necessary information with tags. There already exists the `[system]`, `[user]`, and `[assistant]` tags. So this would expand that to `[vision]`, `[hearing]`, `[tasting]`, etc. 

This is a good start for the architecture of the model, however, this is an overdesigned architecture. After doing some more research, I discovered that most models released today are multimodal, meaning that they can inherently understand different media types baked into their weights. This simplified the project significantly and meant that I didn't need to have this group chat architecture anymore.

## Evolution of the Architecture (Multimodality)
However, this brought up a new issue. I didn't want the model to think that it was executing a request from a user and the images that have been attached are adding context to a given request. I tried to bypass this by attaching the images to the transcript using the `[assistant]` tags, but this confused the model significantly. The fix ended up being just messing with the system prompt.

The next issue arose when I started running out of context window length. Since the model was being inferenced every time it finished its previous output, it ate up tokens very quickly. The fix for this problem was twofold:

1. **Stop sending high-quality images:** I truncated the images from 1080p to 300x300. This is enough to give enough detail to the model, but not too little that it can't understand what is going on anymore.

Testing this was good as now I could have longer conversations with the model about what was happening in real time. 

## Managing Context and the Object Permanence Problem
Using this new architecture was good, however, the model could not understand movement. Since I'm only ever handing it one frame at a time on each inference step, it doesn't actually understand what has happened between those inferences, nor the fact that my hand may have moved out of frame. From its perspective, I was present and then I wasn't. The lack of object permanence could have been a limitation of the model I was using being very small (since I don't have the compute for a larger model), or it could have been the lack of movement data. 

To add movement data but still have enough compute for fast inference times, I took a series of images and sent that series to the model. This actually helped the object permanence issues significantly but also ballooned the token usage again, and within 10 inference steps, **20%** of the available context window was used, which translates to ~3 minutes of real-world use.

The fix for that problem was to stitch together the movement images into one long image and send that new image to the model—essentially a panoramic image but displaying a scene over time. Since I didn't have a lot of compute to send really long strings of images, I set the maximum to 2fps, so 2 images will represent 1 second. These images are also gathered between inference loops so that it always has the maximum available context for the setting.

## Programmatic Evaluation for Inference Efficiency
The second big saving is programmatic evaluation. Going back to thinking as a human, when you are walking in a new area, you will take a couple of seconds to analyze where you are and other environmental factors; however, you won't really "acknowledge" anything else unless it changes. Something that doesn't move doesn't need the same level of focus the third time as it did the first. 

This can be applied to the AI to reduce the amount of times that you inference it by comparing the last seen image to the AI with the current seen image through the camera. If they are more than **X%** different, then inference the LLM. I found success by setting this value to **7.5%**, but it can be increased or decreased.

All of these settings worked to help with the biggest problem facing this kind of architecture and AGI: the context window. These allowed me to go from only being able to interact with this LLM for 10 minutes to being able to interact for hours without running out of context space. This architecture and thinking can be expanded to other senses for multimodal LLMs.

---

## The Concept of "Sleep" for AI Memory Management

*This section continues talking about how to save tokens and context space.*

As a human, when you interact with someone, you will remember what they say immediately with relatively good accuracy, but you will not get it correct with **100%** accuracy. LLMs do not have this limitation; whatever you say to an LLM will be remembered as long as it's in the context window. This led me to think about why that is, and it led me even further down a train of thought. 

If you told me that it was your birthday on the 5th of June, I will remember for that day. However, if I don't reiterate that in my mind or write it down somewhere, then the chances that I remember the date will be lower the following day. I might remember *June* or *5th*, but not both of them. As time goes on and this data is not reinforced into my brain, I forget completely that value. 

When you are awake for too long, you get tired. There is still a lot of research going into why most animals sleep, but in my mind, it's so that you can compress your internal context and learn from that. If you have been awake for 48 hours, you actually remember quite a lot of things; however, your motor functions and other filters will degrade. You might offend someone or leak something important when tired. This closely resembles AI chat degradation over time. 

I believe that sleep works in animals and humans to essentially distill the day's events and then add a heavily compressed version of the previous day's events at the beginning of your context window, capped at `X` "tokens" so that you can remember most of what happened the day before, and the day before that, but the further back your memory goes, the worse it gets.

### Trauma, Emotion, and LoRA
This, however, does not account for life-changing events. If you have a traumatic experience—for example, having a near miss with a vehicle—you don't forget that event. In fact, even years later, you can still remember it almost as if it was just happening for the first time. This is not exclusive to trauma either; I've found that in my life, situations with very high emotions stick around for a lot longer. These experiences change the way you interact with the world too. 

This is reminiscent of LoRA training on an LLM: baking into the weights of an LLM or a person that event to make sure that they don't let that happen again for negative experiences (negative reinforcement training) or seek similar experiences again for positive experiences (positive reinforcement training).

### Repetition and Vector Databases
Another way that humans remember things is through repetitive action. If I study for a maths exam and every day I study the quadratic formula, eventually I'll remember the quadratic formula very well. Does this mean that it's being affected by a human version of LoRA? I don't think so. Modern models can be linked with vector databases, and I believe that humans and other animals have similar mechanics. 

By repeating a piece of knowledge over and over, you build a strong connection to that memory that then can be triggered by bringing it up, or if it's very strong, you can just recall it off the top of your head. However, it will still have to be brought up, unless you are actively thinking about it (it's in your active context window). This also explains why people forget things shortly after an exam has taken place. They would have created a very strong connection to those memories and knowledge, but if it's not constantly used, those connections degrade and then become weaker. Over years and years, that knowledge and memory is forgotten. I believe this also happens during the sleep cycle and is part of why it takes so long.

### Hallucinations and Sleep Deprivation
I also believe AI hallucination is part of this. When I've been awake for multiple days, I genuinely start to hear things and see things and make things up. Unless it's pointed out to me that it's a hallucination, then I wouldn't really recognize it, even if the next day I can recognize that it was a hallucination. It's very similar to how a model can tell you something confidently and be wrong about it and not realize until you correct it.

### Implementing AI Sleep
All of this to say, sleep is very, very important. But how do you actually implement this into an AI architecture? 

1. **Triggering Fatigue:** First, there needs to be a way for the model to feel "tired." This could be when it hits **70%** of its context window or another trigger. Then it'll finish its task and go to "sleep." 
2. **Recursive Analysis:** During sleep, another model of comparable intelligence (think Claude Opus 4.8 vs. Claude Opus 4.7) will recursively analyze the session's context and adjust all the different values that I've spoken about up until this point. 
3. **Waking Up:** Once it's done, a new instance of the brain will be inferenced with the compressed/distilled output of the sleeping model as its first message. 
4. **Interruption:** If this process is interrupted, then that context fails to be wiped fully, and the context window starts already **30%** full instead of **1%** full or even less. This could be why humans feel groggy or snappy when they haven't slept, since they are waking up already over-stimulated, and you have to focus on new inputs while an old system still hasn't finished executing.

## Tools vs. Native Understanding
This is something else that I thought a lot about. You can train an AI model on a lot of things, but tools still exist. I asked Gemini to describe to me how it interprets audio vs. a Google search. This was a while ago so I don't remember the exact transcript, consolidating my previous observations, but the main takeaway I had was that anything that will act in the same way and has constant parameters can be trained on, but real-time data and always-changing variables will always be a tool. 

In this example, audio, vision, speaking, etc., have constant variables and they act in mostly the same way, meaning they can be trained on. However, the results of a Google search are always changing, meaning that you can't train a model to natively understand or know a Google search. Models, just like humans, can find patterns within the results of tool calls, but tools by design cannot be integrated into the weights of a model.

## Finishing Comments
These are all of things that I've thought about and will have to consider when implementing my LLLLM. I'll try to keep this updated as time goes on but I'm not making any promises.