import time
import smtplib
from email.mime.text import MIMEText
def timeprofile(fun):
    def tmp(*args, **kwargs):
        s = time.time()
        r = fun(*args, **kwargs)
        print(fun.__name__ + " " + str(time.time() - s))
        return r
    return tmp
def emailto(task_name):
    text = "Task " + task_name + " completed."
    msg = MIMEText(text, 'plain', 'utf-8')
    smtp = smtplib.SMTP()
    smtp.connect("smtp.126.com")
    smtp.login("guolipengyeah", "217000mh")
    smtp.sendmail("guolipengyeah@126.com", ["guolipengyeah@126.com"], msg.as_string())
    smtp.quit()

    