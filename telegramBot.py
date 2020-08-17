import telegram

TOKEN = '776293533:AAE6f_Ob6XCccZoEyVh0dr0_oX0Uxdy--I4'

class TelegramBot:
    def __init__(self, token_file=None):
        
        if token_file is not None:
            self.token = load_file(token_file).decode("utf-8")
        else:
            self.token = TOKEN
            
        self.bot = telegram.Bot(token=self.token)
        
    def load_file(file):
        with open(file, 'r') as f:
            res = f.read()
        return res
    
    def send(self, message, chat_id=335091033):
        self.bot.sendMessage(chat_id=chat_id, text=message)