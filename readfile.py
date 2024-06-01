import pandas as pd

def process_and_update_csv(filepath,user_query_cmd,status):
    # Load the CSV file
    try:
        data = pd.read_csv(filepath)
        print("File read successfully.")
        if 'user_query_cmd' not in locals() or user_query_cmd is None:
            raise Exception("cmd_now is not defined or is None")  # Raising a generic exception
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Check and combine columns

    if 'prompteg' in data.columns and 'promptold' in data.columns:



        # Combine the columns into a new column 'promptupd'
        #data['promptupd'] = data['prompteg'].astype(str) + " " + data['promptold'].astype(str)
        data.loc[0, 'promptupd'] = data.loc[0, 'prompteg'] + "\n" + user_query_cmd+ ' =>'

        print("Columns combined successfully.")
        prompt_upd_str=data.loc[0, 'promptupd']
        print('1st data:'+prompt_upd_str)

        #print(print(data.loc[1:1, 'promptupd']))


    else:
        print("Required columns are not present in the file.")
        return

    # Save the updated DataFrame back to the same CSV file
    try:
        data.to_csv(filepath, index=False)
        print(f"File updated and saved successfully to {filepath}.")
    except Exception as e:
        status=1
        print(f"Error writing the file: {e}")
    finally:
        print(status)
        print('user query command is saved.')
        #file.close()

# Specify the path to your CSV file
status=0

#filepath = './input-txt.csv'

# Call the function with the file path
#user_query_set='He want to find Mental Health service nearby 19123 =>'
#status=process_and_update_csv(filepath,user_query_set,status)


