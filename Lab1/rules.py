from production import IF, AND, THEN, FAIL, OR


NINJA_RULES = (


    IF( AND( '(?x) has no skill',        
             '(?x) has city map' ),
        THEN( '(?x) is a city guide' )),
         
    IF( AND( '(?x) has a local knowledge',        
             '(?x) is a city guide' ),
        THEN( '(?x) a Lonie' )),

    IF( AND( '(?x) has stealthy movements',        
             '(?x) has precise actions' ),
        THEN( '(?x) is a good balancer' )),

    IF( AND( '(?x) has basic training',        
             '(?x) is a good balancer' ),
        THEN( '(?x) a Genin' )),

    IF( AND( '(?x) has a team leader',        
             '(?x) has precise actions' ),
        THEN( '(?x) is a calm person' )),      

    IF( AND( '(?x) has mission skills',       
             '(?x) is a calm person' ),
        THEN( '(?x) a Chunin' )),  

    IF( AND( '(?x) has nature link',        
             '(?x) has great wisdom' ),
        THEN( '(?x) is a upper level' )),

    IF( AND( '(?x) has mission skills',
             '(?x) is a upper level'),
        THEN( '(?x) a Jonin' )),  

    IF( OR( '(?x) has multiple jutsu' ,
             '(?x) has ultimate jutsu' ),
        THEN( '(?x) is a skilled ninja' )),

    IF( AND( '(?x) has strong mental',
             '(?x) is a skilled ninja'),
        THEN( '(?x) a Sage' )),

    IF( AND( '(?x) has written book',
             '(?x) has strong mental'),
        THEN( '(?x) is a veteran' )),

    IF( AND( '(?x) has a cloack',
             '(?x) is a veteran' ),
        THEN( '(?x) a Kage' ))
)


NINJA_DATA = {
    'naruto has written book',
    'naruto has a cloack',
    'naruto has strong mental',
}
