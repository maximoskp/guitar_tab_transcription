import os

go_on = False

k = input('Delete all generated files and starting from scratch? y/n')

if k == 'y' or k == 'Y' or k == 'yes' or k == 'Yes' or k == 'YES':
    go_on = True

if go_on:
    folders2check = ['data/dadaGP_event_parts', 'track_representation_parts']
    for fold in folders2check:
        if os.path.isdir( fold ):
            for f in os.listdir( fold ):
                os.remove( fold + os.sep + f )
        else:
            os.makedirs( fold, exist_ok=True )