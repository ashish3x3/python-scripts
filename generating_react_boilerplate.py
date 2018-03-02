import os
import sys
import shutil
import json

TOP_LEVEL         = [ 'src', 'package.json', 'webpack.config.js' ]
SRC_SUB_STRUCTURE = [ 'js', 'bundle.js', 'index.html' ]
JS_SUB_STRUCTURE  = [ 'components', 'actions', 'constants', 'reducers', 'pages', 'client.js' ]

def create_folder(folder_name, folder_path):
	path = os.path.join(folder_path, folder_name)
	print 'path ',path
	try:
		if os.path.exists(path):
			print 'path exists'
			shutil.rmtree(path)
			curr_path = os.getcwd()
			print 'curr_path ', curr_path
			os.chdir(folder_path)
			print 'changed dir  create_folder ', os.getcwd()
			os.mkdir(folder_name)
			print 'created succesfully ',folder_name
			os.chdir(curr_path)
			print 'changed dir  create_folder revert to curr_dir ', os.getcwd()
		else:	
			os.mkdir(path)
			print 'created succesfully ',folder_name
	except:
		print('error in create folder ')
		sys.exit(0)


def create_file(filename, extension, path):
	try:
		name = filename + extension
		file = open(os.path.join(path, name), 'w')

		file.close()
	except:
		print('error in create file ')
		sys.exit(0)

def create_top_level_structure():
	for name in TOP_LEVEL:
		print('name in top level ',name)
		if '.' in name:
			file_name,file_extension = os.path.splitext(name)
			print 'file_name,file_extension ',file_name,file_extension
			create_file(file_name, file_extension, os.getcwd())
		else:
			create_folder(name, os.getcwd())

def create_src_structure():
	print('current dir in  create_src_structure ',os.getcwd())
	path = os.path.join(os.getcwd(), 'src')
	print('changed dir in  create_src_structure ',path)

	for name in SRC_SUB_STRUCTURE:
		if '.' in name:
			file_name,file_extension = os.path.splitext(name)
			create_file(file_name, file_extension, path)
		else:
			create_folder(name, path)



def create_js_structure():
	print('current dir in  create_js_structure ',os.getcwd())
	path = os.path.join(os.getcwd(), 'src/js')
	print('changed dir in  create_js_structure ',path)

	for name in JS_SUB_STRUCTURE:
		if '.' in name:
			file_name,file_extension = os.path.splitext(name)
			create_file(file_name, file_extension, path)
		else:
			create_folder(name, path)


def write_default_package_json():
	file_path_def = raw_input('Enter path of package.json: ') or "A:/React-Apps/"
	#reading json
	file_path_read =  os.path.join(file_path_def, 'pack.json')

	with open(file_path_read) as data_file:    
    		data = json.load(data_file)

    # print('data ',data)
    #writing json
	file_path =  os.path.join(os.getcwd(), 'package.json')
	with open(file_path, 'w') as outfile:
			json.dump(data, outfile, indent=4, sort_keys=True, separators=(',', ':')) #json.dump(data, outfile) #




def write_default_webpack_config():
	file_path_def = raw_input('Enter path of package.json: ') or "A:/React-Apps/"
	#reading json
	file_path_read =  os.path.join(file_path_def, 'webpack_default.js')
	read = ''
	f = open(file_path_read, 'rU')
	read = read + f.read() + '\n'
	f.close()

	file_path =  os.path.join(os.getcwd(), 'webpack.config.js')
	f = open(file_path, 'w')
	f.write(read)
	f.close()
	


def create_project_folder(project_name, project_path):
	print('current dir ',os.getcwd())
	os.chdir(project_path)
	print('changed dir ',os.getcwd())
	create_folder(project_name,os.getcwd())
	os.chdir(os.path.join(project_path, project_name))
	print('changed dir ',os.getcwd())
	create_top_level_structure()
	create_src_structure()
	create_js_structure()
	write_default_package_json()
	write_default_webpack_config()

if __name__ == "__main__":
	project_name = raw_input('Enter name of project: ')
	project_path = raw_input('Enter path of project: ') or "A:/React-Apps"
	print 'project_path ',project_path
	if project_path == None or project_path == '':
		print 'default path = A:/React-Apps'
		# project_path = 'A:/React-Apps'
	create_project_folder(project_name, project_path)
	print '#################################################'
	print 'reactApp boilerplate created succesfully'
	print '  -',project_name
	print '    -','src'
	print '       -','js'
	print '          -','components'
	print '          -','actions'
	print '          -','constants'
	print '          -','reducers'
	print '          -','pages'
	print '          -','client.js'
	print '       -','bundle.js'
	print '       -','index.html'
	print '    -','package.json'
	print '    -','webpack.config.js'

