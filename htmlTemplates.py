css = '''
<style>
.chat-message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1rem;
}

.chat-message.user {
    justify-content: flex-start; /* Align user message to the left */
}

.chat-message.bot {
    justify-content: flex-start; /* Align bot message to the left */
}

.chat-message .avatar {
    font-size: 2rem; /* Size of the emoji */
    margin-right: 1rem; /* Space between the emoji and the message box */
}

.chat-message .message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    background-color: #2b313e;
    color: #fff;
    flex-grow: 1; /* Allows the message box to take up the remaining space */
}

.chat-message.bot .message {
    background-color: #475063; /* Different background for bot message */
}
</style>

'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">👤</div>    
    <div class="message">{{MSG}}</div>
</div>
'''
