import smtplib 
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
# start TLS for security 
s.starttls()   
# Authentication 
s.login("sender's email", "pass") 
# message to be sent 
message = "Model accuracy is above 80%"
# sending the mail 
s.sendmail("sender mail", "reciever mail", message) 
# terminating the session 
s.quit()
