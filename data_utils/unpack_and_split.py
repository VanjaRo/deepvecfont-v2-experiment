import os
import zipfile
import shutil

"""
Usage instructions:

python3 unpack_and_split.py

- run this script from a folder containining archives of the fonts
- it will extract font files into tmp folder 
- then it will separate the content into 2 folders "roman" and "italic"
- only fonts with pairs are going to be moved
- font files with no pair will be deleted after each archive iterarion

- folders "roman" and "italic" are then are being archived 
- and placed in the directorry the script was called from
"""


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


def is_italic(filename):
    """Checks if the filename indicates an italic font."""
    return filename.endswith("italic") or filename.endswith("oblique")


def move_font_pair(source_dir, roman_dir, italic_dir, style_sep=" "):
    """Moves font pair (roman and italic) to designated directories."""
    all_fonts = os.listdir(source_dir)
    lowered_to_normal = {}
    for i in range(len(all_fonts)):
        lowered = all_fonts[i].lower()
        lowered_to_normal[lowered] = all_fonts[i]
        all_fonts[i] = lowered

    font_pairs = {}
    font_ext = {}
    for name in all_fonts:
        # get rid of the extencion
        name_split_dot = name.split(".")
        if len(name_split_dot) < 2:
            continue
        name, ext = name_split_dot[0], name_split_dot[1]
        font_ext[name] = ext

        companion_name = None
        if is_italic(name):
            # remove style word and search for the roman companion
            name_split_sep = name.split(style_sep)
            name_no_italic = f"{style_sep}".join(name_split_sep[:-1])
            name_regular = name_no_italic + style_sep + "regular"
            if name_no_italic in font_pairs:
                companion_name = name_no_italic
            elif name_regular in font_pairs:
                companion_name = name_regular

        else:
            name_cmp = name
            if name.endswith("regular"):
                name_cmp = f"{style_sep}".join(name.split(style_sep)[:-1])
            name_italic, name_oblique = (
                name_cmp + style_sep + "italic",
                name_cmp + style_sep + "oblique",
            )
            if name_italic in font_pairs:
                companion_name = name_italic
            elif name_oblique in font_pairs:
                companion_name = name_oblique

        if companion_name is None:
            font_pairs[name] = True
        else:
            name_with_ext = lowered_to_normal[name + "." + font_ext[name]]
            companion_with_ext = lowered_to_normal[
                companion_name + "." + font_ext[companion_name]
            ]
            roman_file = os.path.join(source_dir, name_with_ext)
            italic_file = os.path.join(source_dir, companion_with_ext)
            if is_italic(name):
                roman_file, italic_file = italic_file, roman_file

            os.rename(roman_file, os.path.join(roman_dir, os.path.basename(roman_file)))
            os.rename(
                italic_file, os.path.join(italic_dir, os.path.basename(roman_file))
            )
    delete_files_in_directory(source_dir)


def process_archive(archive_dir, archive_name, roman_dir, italic_dir):
    """Unzips archive and moves font pairs."""
    tmp_dir = os.path.join(archive_dir, "tmp")
    if not (os.path.exists(tmp_dir) and os.path.isdir(tmp_dir)):
        os.mkdir(os.path.join(archive_dir, "tmp"))

    archive_path = os.path.join(archive_dir, archive_name)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)  # Extract to tmp directory

    move_font_pair(tmp_dir, roman_dir, italic_dir)


def main():
    """Main function to iterate through archives and process them."""
    archive_dir = os.getcwd()  # Assuming script is in the directory with archives
    roman_dir = os.path.join(archive_dir, "roman")
    italic_dir = os.path.join(archive_dir, "italic")
    result_archive_name = "italic_roman"

    # Create destination directories if they don't exist
    if not os.path.exists(roman_dir):
        os.makedirs(roman_dir)
    if not os.path.exists(italic_dir):
        os.makedirs(italic_dir)

    for filename in os.listdir(archive_dir):
        if filename.endswith(".zip") and not filename == result_archive_name + ".zip":
            process_archive(archive_dir, filename, roman_dir, italic_dir)

    # archive italic and roman folders
    result_archive_name = "italic_roman"
    italic_roman_arch = os.path.join(archive_dir, result_archive_name)
    os.makedirs(italic_roman_arch)
    os.rename(roman_dir, os.path.join(italic_roman_arch, os.path.basename(roman_dir)))
    os.rename(italic_dir, os.path.join(italic_roman_arch, os.path.basename(italic_dir)))

    shutil.make_archive(result_archive_name, "zip", italic_roman_arch)
    shutil.rmtree(italic_roman_arch)

    # with zipfile.ZipFile(italic_roman_arch, "w") as zip_ref:
    #     zip_ref.write(os.path.basename(roman_dir))
    #     zip_ref.write(os.path.basename(italic_dir))


if __name__ == "__main__":
    main()
