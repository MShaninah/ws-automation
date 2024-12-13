import configparser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import tkinter as tk
from tkinter import messagebox
import threading
from selenium.webdriver.common.action_chains import ActionChains
from datetime import datetime, timedelta


config = configparser.ConfigParser()
config.read('config.ini')


edge_driver_path = config['settings']['edge_driver_path']
pdf_file_path = config['settings']['pdf_file_path']


options = webdriver.EdgeOptions()
options.add_argument('start-maximized')
options.add_argument("--disable-extensions")
options.add_argument("--disable-infobars")
options.add_argument("--disable-notifications")


service = Service(executable_path=edge_driver_path)
driver = webdriver.Edge(service=service, options=options)


driver.get("https://purchasingprogramsaudi.com/Index.cfm")

wait = WebDriverWait(driver, 20)
driver.implicitly_wait(2)


def get_view_buttons():
    try:
        view_buttons = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[@class='input_btn' and contains(@onclick, 'ColdFusion.navigate')]")))
        return view_buttons
    except Exception:
        return []


def get_remaining_time_from_page():
    try:

        time_element = wait.until(EC.presence_of_element_located((By.XPATH, "//span[@id='remaining_time']")))  
        time_text = time_element.text.strip()


        minutes, seconds = map(int, time_text.split(':'))
        remaining_time = timedelta(minutes=minutes, seconds=seconds)

        print(f"الوقت المتبقي المستخرج من الصفحة: {remaining_time}")
        return remaining_time

    except Exception as e:
        print(f"Error extracting remaining time: {e}")
        return timedelta(seconds=0)  


def check_remaining_time():
    try:
        remaining_time = get_remaining_time_from_page()  

        if remaining_time <= timedelta(seconds=0):
            print("انتهى الوقت! سيتم تحديث الصفحة.")
            driver.refresh()
            return False  
        else:
            minutes, seconds = divmod(remaining_time.seconds, 60)
            print(f"الوقت المتبقي: {minutes} دقيقة و {seconds} ثانية")
            return True 

    except Exception as e:
        print(f"Error checking remaining time: {e}")
        return True  
def initialize_timer():
    global start_time, total_duration, remaining_time
    
    total_duration = timedelta(minutes=15)  
    start_time = datetime.now()
    remaining_time = total_duration
    print(f"تم بدء العد التنازلي من {remaining_time}")

def show_user_selection_ui():
    root = tk.Tk()
    root.title("اختيار المستخدم")
    root.geometry("400x300")
    root.config(bg="#f0f0f0")

    label = tk.Label(root, text="اختر المستخدم:", font=("Arial", 12), bg="#f0f0f0")
    label.pack(pady=20)


    users = {
        "CMM": ("MOH-C1171", "Faa@12345"),
        "CMR": ("MOH-C1151", "Asdf1234567#"),
        "SLM": ("MH-H522099", "AlSalam@2024"),
        "MDD": ("MH-H560364", "Abc123$$")
    }


    def select_user(user_key):
        username, password = users[user_key]
        login_and_show_buttons(username, password)
        root.withdraw() 


    for user_key in users:
        tk.Button(root, text=f"{user_key}", command=lambda user_key=user_key: select_user(user_key), width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 10), relief="solid", bd=1).pack(pady=5)

    root.mainloop()


def login_and_show_buttons(username, password):
    try:

        username_field = wait.until(EC.presence_of_element_located((By.ID, "j_username"))).send_keys(username)
        password_field = driver.find_element(By.ID, "j_password")
        password_field.send_keys(password)

        login_button = wait.until(EC.element_to_be_clickable((By.ID, "btnLogin")))

        driver.execute_script("arguments[0].click();", login_button)


        waiting_confirmation_link = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//span[@class='div_assist_dashboard_Title']//a[contains(text(), 'Waiting Confirmation Referral Requests')]"))
        )
        driver.execute_script("arguments[0].click();", waiting_confirmation_link)


        show_buttons_ui()

    except Exception as e:
        print(f"Error during login or navigation: {e}")
        pass


def show_buttons_ui():
    try:
        view_buttons = get_view_buttons()
        if len(view_buttons) == 0:
            messagebox.showinfo("معلومات", "لا توجد أزرار لعرضها.")
            return

        root = tk.Tk()
        root.title("بوت الطلبات")
        root.geometry("300x400")
        root.config(bg="#f0f0f0")

        label = tk.Label(root, text="اختر الطلب:", font=("Arial", 12), bg="#f0f0f0")
        label.pack(pady=10)

        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=20)

        for i, button in enumerate(view_buttons):
            button_name = f"view {i + 1}"
            tk.Button(button_frame, text=button_name, command=lambda i=i: execute_button_action(i, root), width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 10), relief="solid", bd=1).pack(pady=5)


        def exit_program():
            try:
                driver.quit()
            except:
                pass
            root.quit()
            print("تم إغلاق البرنامج بالكامل.")

        exit_button = tk.Button(root, text="خروج", command=exit_program, width=20, height=2, bg="#f44336", fg="white", font=("Arial", 10), relief="solid", bd=1)
        exit_button.pack(pady=10)

        threading.Thread(target=monitor_remaining_time, daemon=True).start()

        root.mainloop()

    except Exception as e:
        print(f"Error during Tkinter UI: {e}")
        pass

def monitor_remaining_time():
    while True:
        time.sleep(2) 
        if not check_remaining_time():
            break  


def execute_button_action(button_index, root):
    try:
        view_buttons = get_view_buttons()
        if button_index < len(view_buttons):

            view_buttons[button_index].click()

   
            link_documents = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Link documents')]")))
            driver.execute_script("arguments[0].click();", link_documents)


            all_windows = driver.window_handles
            main_window = driver.current_window_handle
            for window in all_windows:
                if window != main_window:
                    driver.switch_to.window(window)
                    break


            title_field = wait.until(EC.presence_of_element_located((By.ID, "Title")))
            title_field.send_keys("A")

            file_input = wait.until(EC.presence_of_element_located((By.ID, "file1")))
            current_value = file_input.get_attribute('value')
            if not current_value:
                file_input.send_keys(pdf_file_path) 


            dropdown = wait.until(EC.visibility_of_element_located((By.ID, "Type1")))
            select = Select(dropdown)
            select.select_by_value("11")


            submit_button = wait.until(EC.element_to_be_clickable((By.ID, "submit")))
            driver.execute_script("arguments[0].click();", submit_button)


            driver.close()


            driver.switch_to.window(main_window)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


            accept_button = wait.until(EC.element_to_be_clickable((By.ID, "accept")))


            driver.execute_script("arguments[0].click();", accept_button)


        root.withdraw()  

        time.sleep(8)
        root.deiconify()  

    except Exception as e:
        print(f"Error during button action: {e}")
        pass

if __name__ == "__main__":

    show_user_selection_ui()
