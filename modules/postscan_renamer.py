import glob, os

retval = os.getcwd()
print("\nCurrent working directory is:\n %s" % retval)
iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
os.chdir(iris_path.strip('"'))

all_files = sorted(glob.glob('*.*'))
print("There are " + str(len(all_files)) + " files.")
files_to_rename = [file for file in all_files if len(file.split(".")) > 3]
mirror_file = str(glob.glob('*000.pgm')).strip("'[]'")
if mirror_file:
    files_to_rename.remove(mirror_file)
print(files_to_rename)
toggle = input("OK to rename " + str(len(files_to_rename)) + " files? (y/[n])\t")
if toggle not in ("n", "no"):
    new_number = input("What scan number to set files to? (Enter a number between 1 and 999)\t")
    new_names = []
    for file in files_to_rename:
        parsed_file = file.split(".")
        if parsed_file[2].isdigit():
            while not new_number.isdigit():
                new_number = str(input("Please enter a NUMBER.\t"))
            else:
                while not (999 > int(new_number) > 1):
                    new_number = str(input("Please enter a VALID number.\t"))
                else:
                    parsed_file[2] = ('0' * (3 - len(new_number))) + new_number
            renamed_file = ".".join(parsed_file)
            new_names.append(renamed_file)
    print(new_names)
    i = 0
    for x in files_to_rename:
        os.rename(files_to_rename[i],new_names[i])
        i += 1

else: exit()





# txt_list = sorted(glob.glob('*.txt'))
# pgm_list = sorted(glob.glob('*.pgm'))
# csv_list = sorted(glob.glob('*.csv'))
# #xml_list = sorted(glob.glob('*/*.xml'))
# #if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))
# chip_name = pgm_list[0].split(".")[0]
# mirror_file = str(glob.glob('*000.pgm')).strip("'[]'")
# if mirror_file:
#     pgm_list.remove(mirror_file)
#     print("Mirror file present")
# else: print("Mirror file absent") #mirror_toggle = False
# fluor_files = sorted(glob.glob('*.A.pgm'))
# if fluor_files:
#     [pgm_list.remove(file) for file in pgm_list if file in fluor_files]
#     print("Fluoresent images detected")
