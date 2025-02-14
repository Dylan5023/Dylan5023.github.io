---
layout: single
title:  "Movie_textmining"
categories: ML
tag: [python, blog, jekyll, matplotlib, seaborn, ML]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

**[Notice]** [ML_practical practice_3]
{: .notice--info}

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 1) Library & Data Import



```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/bourne_scenario.csv")
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page_no</th>
      <th>scene_title</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1         EXT. MERCEDES WINDSHIELD -- DUSK</td>
      <td>1                It's raining...             ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A1        INT. MERCEDES -- NIGHT</td>
      <td>A1                On his knee -- a syringe an...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2         INT. COTTAGE BEDROOM -- NIGHT</td>
      <td>2                BOURNE'S EYES OPEN! -- panic...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A2        INT. COTTAGE LIVING AREA/BATHROOM ...</td>
      <td>A2                BOURNE moving for the medic...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3         INT./EXT. COTTAGE LIVING ROOM/VERA...</td>
      <td>3                One minute later.  BOURNE mo...</td>
    </tr>
  </tbody>
</table>
</div>


-----


## 2) Explore the dataset


### 2-1) Explore basic information



```python
df.shape
```

<pre>
(320, 3)
</pre>

```python
df.isnull().sum()
```

<pre>
page_no        0
scene_title    0
text           0
dtype: int64
</pre>

```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 320 entries, 0 to 319
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   page_no      320 non-null    int64 
 1   scene_title  320 non-null    object
 2   text         320 non-null    object
dtypes: int64(1), object(2)
memory usage: 7.6+ KB
</pre>

```python
df['text'][0]
```

<pre>
" 1                It's raining...                Light strobes across the wet glass at a rhythmic pace...                 Suddenly -- through the window a face -- JASON BOURNE --               riding in the backseat -- his gaze fixed.      "
</pre>
-----


## 3) Text data preprocessing


### 3-1) Apply regular expressions



```python
import re

def apply_regular_expression(text):
    text = text.lower()
    english = re.compile('[^ a-z]')
    result = english.sub('', text)
    result = re.sub(' +', ' ', result)
    return result
```


```python
df['processed_text'] = df['text'].apply(lambda x: apply_regular_expression(x))
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page_no</th>
      <th>scene_title</th>
      <th>text</th>
      <th>processed_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1         EXT. MERCEDES WINDSHIELD -- DUSK</td>
      <td>1                It's raining...             ...</td>
      <td>its raining light strobes across the wet glas...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A1        INT. MERCEDES -- NIGHT</td>
      <td>A1                On his knee -- a syringe an...</td>
      <td>a on his knee a syringe and a gun the eyes of...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2         INT. COTTAGE BEDROOM -- NIGHT</td>
      <td>2                BOURNE'S EYES OPEN! -- panic...</td>
      <td>bournes eyes open panicked gasping trying to ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A2        INT. COTTAGE LIVING AREA/BATHROOM ...</td>
      <td>A2                BOURNE moving for the medic...</td>
      <td>a bourne moving for the medicine cabinet digs...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3         INT./EXT. COTTAGE LIVING ROOM/VERA...</td>
      <td>3                One minute later.  BOURNE mo...</td>
      <td>one minute later bourne moves out onto the ve...</td>
    </tr>
  </tbody>
</table>
</div>


### 3-2) Word Count


##### Create a corpus (corpus)



```python
# make corpus
corpus = df['processed_text'].tolist()
corpus
```

<pre>
[' its raining light strobes across the wet glass at a rhythmic pace suddenly through the window a face jason bourne riding in the backseat his gaze fixed ',
 ' a on his knee a syringe and a gun the eyes of the driver jarda watching bournes pov the passenger back of his head cell phone rings the head turns its conklin bourne returns his stare ',
 ' bournes eyes open panicked gasping trying to stay quiet marie sleeps ',
 ' a bourne moving for the medicine cabinet digs through the medicine cabinet downs something specific ',
 ' one minute later bourne moves out onto the veranda marie pads in watching him for a moment concerned clearly its not the first time this has happened they both look different than last we saw them his hair is longer shes a blonde hippie travelers their cottage is humble but sweet the bedroom opens to a beach and a town just down the hill club music from some all night rave wafting in from the far distance marie where were you jason bourne in the car conklin up front marie ill get the book bourne no theres nothing new marie youre sure he nods we should still we should write it down bourne two years were scribbling in a notebook marie it hasnt been two years bourne its always bad and its never anything but bits and pieces anyway shes gone quiet you ever think that maybe its just making it worse you dont wonder that she lays her hands on his shoulders steadies him marie we write them down because sooner or later youre going to remember something good bourne softens i do remember something good all the time i remember you she smiles kisses him leads him back in ',
 ' marie getting bourne into the bed turning down the light getting him settled waiting for that pill to kick in what would he do without her bourne im trying marie okay marie i worry when you get like this bourne its just a nightmare marie i dont mean that i worry when you try to ignore it he hesitates but that gets him he knows shes right and with that opening hes letting go resistance folding almost childlike shes gathering him in hes letting her do it marie contd sleep sleep now bourne i should be better by now marie you are better and i think its not memories at all its just a dream you keep having over and over bourne but it ends up the same marie one day it will be different it just takes time beat well make new memories you and me silence she strokes his face he gives in to her tenderness hes fading two waifs in the dark ',
 ' bourne running in the sun a punishing pace along the sand moving strong effortless deep into it focused the stunning conjunction of sun and scenery are lost on him ',
 ' a busy market town fishing town hippie town lots of young western faces rundown and happening at the same time marie shopping filling a bag with local produce ',
 ' bourne still running leaving the beach behind ',
 ' marie back from the market putting the groceries away almost done when she stops for a moment a photograph there on the windowsill a snapshot jason and marie on a beach her arms around him as if she were the protector big smiles young alive in love marie smiles ',
 ' funky busy colonial facades in vivid subcontinental technicolor loud morning traffic camera finds bourne coming out of a store with a big bottle of water hes just finished his run standing there chugging away checking the scene when something catches his eye his pov the street a silver car something newish pulling down the block cant quite see whos driving but back to bourne watching this silver car so serious hes casual nobody passing would notice but we do hes on alert moving with him as bourne follows the silver car on foot natural cruising the busy sidewalk blending into the mix chugging on that water bottle and up ahead the silver car making the corner and turning now back to bourne slowing as he reaches the corner his pov the silver car has parked theres a guy welldressed casual physical sunglasses call him kirill hes out of the car and heading across the street toward a building there a telegraph office back to bourne checking his watch the car the guy perimeter ',
 ' mr mohan at his desk hes a crisp proper man of fifty hes just been handed something a photograph of marie an old passport picture mr mohan and your question sir kirill across the desk kirill shes my sister theres been a death in the family this is the last place we know she called from ',
 ' a note on the table im at the beach bourne has just come in just read the note balling it quickly in fact everything is quickly now because bourne is bailing fast calm methodical some exfil procedure that hes honed and choreographed packing like a machine rapid time cuts backpacks thrown open on the bed house cash pulled from a lamp base credit cards taped under the counter ',
 ' kirill coming out of the bank mission accomplished heading back to the silver car getting in and ',
 ' kirill starting it up glancing around nice and easy hes cool putting the car into gear he makes a slow pass through the marketplace eyes everywhere ',
 ' bourne done the place is stripped pulling on the backpacks glancing around one last thing shit he almost missed it the photograph the one of he and marie on the beach the one we saw her looking at earlier there it is on the windowsill jamming it into his pocket and ',
 ' a kirill now parked and out of the car on the move on foot he begins a sweep of the beach ',
 ' bourne out the back jogging keeping low into the neighborhood through the alleys nothing random about it this has all been worked out and ',
 ' crowded with tourists sunbathers marie at her favorite spot talking with two women laughing with them happy ',
 ' a a burly jeep comes roaring up bourne spots the silver car parks at the other end takes off towards the beach ',
 ' kirill methodically making his way up the beach checking every blue tent every towel ',
 ' bourne coming up the beach the opposite way one eye on kirill one eye on marie he arrives just as kirill looks up and sees them a hundred yards away a hard stare between them bourne bends down bourne we gotta go marie we gotta go now from the tone of his voice she knows its serious marie grabs her bag a quick goodbye to the friends they hurry off bourne uses the sunbathers as cover kirill retreats ',
 ' they reach the jeep she knows the drill bag tossed in the back even as the jeep pulls away and ',
 ' bourne driving marie beside him bourne were blown she hesitates one minute ago everything was fine marie no how bourne the telegraph office marie but we were so careful bourne we pushed it we got lazy ',
 ' kirill already back at the silver car following them out onto the main street blocked by the local traffic pulling a huge automatic pistol out from his travel bag ',
 ' the jeep pulling down this narrow little passageway and bournes windshield pov main street packed with traffic and back to bourne not liking this eyes all over trying to decide marie but youre sure bourne he was at the campground yesterday marie so bourne its wrong guy with a rental car and hundred dollar sneakers sleeps in a tent trying to decide whether to pull out or back up marie thats crazy bourne no not this this is real suddenly and hes right there throwing the car into reverse marie where bourne back there at the corner hyundai silver ',
 ' kirill trapped in some main street gridlock glancing back for a way out freezing suddenly because there his pov the jeep the alley right there twenty yards back a good look at bourne and marie as they disappear and ',
 ' the jeep backing up the way it came blowing its horn because an old van pulls in and blocks him from behind ',
 ' bourne leaning on the horn shit now theyve got to wait marie but youre not youre not sure bourne we cant wait to be sure marie i dont want to move againi like it here bourne look we clear out we get to the shack we get safe we hang there awhile ill come back ill check it out but right now we cant marie wheres left to go bourne theres places we cant afford to be wrong ',
 ' kirill calm possessed of a familiar tactical patience he cant get the hyundai to the alley from where he is and it doesnt make sense to go on foot he checks his rearview fuck it theres an opening ahead and hes taking it even though its away from them hell find another way ',
 ' bourne sees the hyundai move forward into traffic the old van is still blocking them from behind bourne you drive marie what bourne already squeezing over switch you drive marie where bourne make the left toward the bridge marie scrambling over the seat bourne eyes everywhere checks his watch the jeep squirts back on the main street and ',
 ' marie at the wheel adrenaline pumping clear running for thirty yards ahead and marie skidding them into the right turn clipping another vehicle mirror shattering speeding up bourne scanning behind them marie moving out to pass veering back an oncoming bus just in time and marie jesus glancing over is he back there bourne not yet marie its just him bourne yeah one guy i dont think he was ready marie hang on marie bearing down pulling out gives him a quick smile bourne knowing hes got a good one here ',
 ' kirill stopping short on a rise bit of a view from here gets half out the car to look below the jeep headed for a bridge hes gonna lose them kirills mind racing grabs duffle from the back abandons car ',
 ' marie driving bourne preps his pistol eye out for kirill bourne you keep going to the shack ill meet you there in an hour marie concerned where are you going bourne im going to bail on the other side and wait this bridge is the only way he can follow marie what if its not who you think it is bourne if he crosses the bridge it is marie there must be another way bourne i warned them marie i told them to leave us alone marie jason please dont do thisit wont ever be over like this bourne theres no choice her pov the old concrete bridge ahead almost there ',
 ' kirill slams into it quick precise grabs into the bag only a moment and hes got a sniper rifle ',
 ' a bourne pistol in hand spare clip in the other checks his watch bourne at the end make the left when i roll out do not slow down marie nods got it after a beat marie i love you too bourne tell me later marie looks ahead ',
 ' b kirill eye to the scope sniper scope pov there the jeep rumbling across the bridge no clear target just the back of the full drivers side headrest kirills finger squeezing firing ',
 ' the jeep jerking front fender tearing into and along the guard rail cement shards fill the air bourne reaching for the wheel too late as the jeep finally crashes through the flimsy guardrail plummets splashes hard begins to sink out of sight ',
 ' kirill lowers the scope takes a quick look around hes basically gone unnoticed in this little nook with his silenced rifle but people are already rushing toward the bridge then there an old woman looking directly at kirill from a doorway not quite sure what but an old indian woman in goa so what kirill drills her with a look as she sinks back inside ',
 ' swallowed up bourne and marie gone ',
 ' kirill scans the surface of the river under the bridge waiting ',
 ' mud plumes as the jeep settles bourne reaches over to marie tries to urge her out ',
 ' kirill with a killers patience waiting almost done scope pov the surface of the water unbroken kirill scans his perimeter theres the old woman again but more people with her people coming out of the woodwork kirill checks the surface one last time nothing he breaks down the rifle in moments goes ',
 ' bourne up into an air pocket held by the jeeps canvas top a big gulp of air and hes back to marie frantic trying to unclip her seatbelt pull her out but its all jammed up ',
 ' bag chucked in the back all he has left is the scope one last look to the unbroken surface then its time to go kirill drifting away disappears ',
 ' the red halo growing bigger blood bourne pauses maries face is blank shes dead bourne finally pulling back realizing this is goodbye ',
 ' we pick up a man with a briefcase on a telephoto lens teddyradio vo the seller has arrived berlin as the man comes to a chinese restaurant he stops squarely so he can be seen clearly then he enters a stark glass office building teddyradio vo contd contd hes inside ',
 ' two men cross the square to the chinese restaurant vic is forty steelass intel operator he carries a large samples case beside him mike younger exnavyseal ',
 ' the hub secure anonymous office space somewhere in the city shades drawn lots of gear cabled around the stale improvised feel of a temporary outpost four serious people alone in this room pamela landy is a senior cia counterintelligence officer hovering over the communications console cronin pamelas early forties stonecold facade quarterbacking the operation over the radio kurt and kim are the techs here his and her headphones ruggedized laptops and comm gear spread around them cronin what have you got survey one ',
 ' dark teddy at the window another military face radio rig night scope watching vic and mike pass below him teddyradio over hub this is survey one mobile one is in motion seller is inside and waiting ',
 ' vic and mike slow as they come to the same stark glass office building teddyradio over we are ready to go ',
 ' mike and vic shake hands two tired coworkers parting ways mike will keep walking vic entering the building through the big glass doors smiling as hes approached by a night shift security guard and we hear mike still walking alone now heading away from the glass office building toward a van parked up the block mikeradio sleeve mike earpiece this is escort one im clear ',
 ' the command post cronin works the communications board cronin all teams listen up we are standing by for final green turning now to pamela who has been listening just as shes about to give the final word kim raises a finger kim langley she hands pamela a phone thats patched into her board pamela a bit surprised martin ',
 ' three men cia mandarins sit around a round table martin marshall deputy vicedirector hes in charge all is tense marshall im here so is donnie and jack weller we understand youre using the full allocation for this buy pamela thats where we came out marshall its a lot of money pam pamela were talking raw unprocessed kgb files its not something we can go out and comparison shop marshall still pamela for a thief a mole i vetted the source marty hes real if it does nothing more than narrow the list of suspects its a bargain at ten times the price mandarin pamela jack weller here its the quality thats at issue pamela yes sir im in total agreement if theyre fakes theyre expensive furious impatient gentlemen ive got the seller on site and in play quite honestly theres not much more to talk about marshall looks to his mandarians not convinced but doesnt want to lose the opportunity time to wash his hands marshall all right pam your game your call ',
 ' all eyes on pamela as she puts down the phone to langley nodding to cronin yes croninradio final green you are go repeat you are go for final green ',
 ' vic has just passed muster with the security guard hes standing alone at an elevator bank vicradio sleeve mike earpiece on my way up vic pulling his earpiece going dark waits for an elevator ',
 ' a dark a small room full of wiring and infrastructure lit by the glare of someones maglight gloved hands quickly pass over racks of gear and wiring and then stopping at the main electrical risers they carefully place an explosive device no bigger than a pack of cigarettes onto the main riser done with that here comes a second small explosive device but this ones special its being taken from a plastic bag and mounted down by the floor on a subpanel done the hands hold up what looks like a piece of tape ',
 ' transferring it onto the charge ',
 ' vic alone with the samples case pressing the button for the top floor the doors close the car rises and then it stops vic bracing himself as the door opens and ivan russian the guy we saw outside with the briefcase standing in an empty darkened hallway ivan show me vic here ivan holding open the door now show now vic flips open the case cash three million dollars ',
 ' a glass door a suite of offices beyond clean anonymous one light on deep inside caspiexpetroleum cherbourg moscow rome tehran',
 ' curtains drawn lights low ivan sitting with the samples case counting the cash vic poring over russian document files dozens of kgb files old and new spread sheets financial data incomprehensibly cyrillic marked up but judging by the seals and clearance sign offs all topsecret vic this is everything ivan is there is all there suddenly music a radio some tinny pop tune just started playing from somewhere down the hall vic what the hell is that alone you said alone both of them sure theyre being doublecrossed vic contd contd reaching for his ankle who who else is here ivan no not me no other people vic coming up with a pistol shut up just shut the freaked by the gun ivan to his feet vic pushing him back as he rushes past the sample case spilling cash and wrong snapph snapph snapph snapph snapph five fast suppressed small caliber shots vic falls first ivan crashing back across a desk as the bullets tear into him both of them dead before they hit the floor and reverse to find the gloved hands unscrewing a silencer tucking away the weapon already in motion before we know whats happened pulling a climbing duffel out from his back pack stuffing in the samples case and ivans briefcase all the files all the money except wait hes left out one old kgb file cover and now he pulls a plastic bag from his backpack gloved hands carefully remove a single sheet of paper from inside the bag and this paper looks exactly like all the stuff hes just tucked away another page full of cyrillic blur hes putting this sheet of paper inside the file cover now hes slipping them both underneath the desk tossing them there as if they fell in the struggle and ',
 ' the electrical risers as one of the two detonation decives blows a single tidy selfcontained explosion and ',
 ' as the lights flicker and fail and the night shift security guard is suddenly cast into darkness and ',
 ' as they were waiting but only a moment before teddyradio sudden urgent hub we just we lost power the building the whole place just went dark cronin looking at pamela the first whiff of dread as cronin repeat who is dark the target building or your location radio voices piling up panicked confusion cascading as ab ',
 ' anonymous drone barn kirill stepping out of a car hes carrying the duffle ',
 ' kirill heading down the hall ',
 ' kirill enters its a small room gretkov is waiting hes forty professional trim and polished dominant gretkov russian youre early kirill youre complaining gretkov its clean kirill would i bring it gretkov taking over now tosses some money on the bed checks out the photocopy of the files gretkov what are you doing kirill stripping quickly kirill im taking a shower its been a long day gretkov make it fast my plane is waiting gretkov dumping three million dollars over the bed as kirill sheds his clothes and we ',
 ' a workmen cluster as a cable winches the jeep is raised from the river bottom as water pours off of it bourne watching from a distance empty ',
 ' b crime scene police blocking office workers from getting in the building media vans clogging the street pamela and cronin across the street watching the mood is black ashes pamela we need to get in there cronin im working on it pamela stands there silent staring at the disaster across the street a ',
 ' a bourne is bailing exfil procedure but this is a heartbroken exfil a footlocker open bournes main stash bourne going through the footlocker setting aside his work clothes other things he needs but he also has to separate a growing pile of marie memories bank cards phony student ids loose passport photos with a mix of looks and hairdos clothes vacuumpacked bags spare shoes ',
 ' b a gasolinestoked fire burning in a rocklined pit bourne feeding his papers and all of maries belongings into the fire a passport cover crinkles back to reveal her photo her face begins to burn gassoaked clothes tossed in nothing left except the photograph the picture of he and marie at the beach the one from his desk bourne hesitates holds the photo out to the flames the rules of exfil say drop it but he cant wont he reaches to his bag sticks the photo on top of his gear then hefting the bag bourne strides away ',
 ' a folding table covered with xeroxed berlin police paperwork pamela getting a showandtell from cronin and teddy cronin so there were two of these explosive charges placed on the power lines one of them failed the fingerprint pamelas got it thats from the one that didnt go off pamela and the germans cant match it teddy nobodys got it we checked every database we could access nothing cronin show her the other thing teddy this is a kgb file that mustve fallen somehow and then slipped under i guess a desk there or handing it to her pamela do we know what this says teddy yup a scrap of paper the main word there the file heading translates as treadstone pamela what the hell is a treadstone cronin shaking his head nobody knows ',
 ' c bourne bouncing around on an old punjab bus alone in a crush of humanity going only god knows where ',
 ' a pamelas pov as she drives toward the entrance cia headquarters virginia ',
 ' a long bright sterile hallway pamela and cronin walking briskly alongside a uniformed sps officer ',
 ' pamela and cronin watching the sps officer unlock the operation panel coding in they begin to descend and ',
 ' drab and desolate pamela and cronin come around a corner walking with a new escort officer passing a sign that reads operations library center ',
 ' sealed triplelocked numbered door it swings open lights flicker on tons of shit packed away in here shelves bulging boxes tapes binders hard drives pamela steps in a huge filing cabinet labeled treadstone pamelaphone over ward abbott os yes pamelaphone pamela landy a ',
 ' ward abbott at his desk the cluttered clubhouse hq of a man whos spent the last thirtyfive years in the spy game a picture window offers a commanders view of the bullpen abbottphone what can i do for you pam pamelaphone i was hoping you had some time for me abbottphone time for what pamelaphone im free right now actually abbottphone that sounds ominous let me check my schedule abbott holds the phone eyes drifting out the window and abbotts pov the bullpen cronin is standing with daniel zorn one of abbotts trusted s clearly zorn is getting the less polite version of pamelas invitation zorn managing to shoot a quick questioning glance to abbott as ',
 ' a cold room desk two chairs abbott and pamela alone pamela treadstone abbott never heard of it pamela thats not gonna fly abbott with all due respect pam i think you mightve wandered a little past your pay grade she has a piece of paper she slides it forward pamela thats a warrant from director marshall granting me unrestricted access to all personnel and materials associated with treadstone abbott rocked and trying to hide it abbott and what are we looking for pamela i want to know about treadstone abbott to know about it almost amused it was a kill squad black on black closed down two years ago more abbott contd nobody wants to know about treadstone not around here the warrant you better take this back to marty and make sure he knows what youre doing pamela trump card he does ive been down to the archives i have the files ward ',
 ' a a hard working port a big mediterranean ferry coming in naples ferry bourne at the rail unchanged from india staring ahead as europe looms ',
 ' b bourne disembarking to an immigration queue looking unremarkable just one of many passing through ',
 ' as they were abbott watching pamela pull a photo from her file sliding it over conklins face peering back pamela lets talk about conklin abbott what are you after pam you want to fry me you want my desk is that it pamela i want to know what happened abbott what happened jason bourne happened fury focusing youve got the files then lets cut the crap it went wrong conklin had these guys wound so tight they were bound to snap more abbott contd bourne was his number one guy went out to work screwed the op and never came back conklin couldnt fix it couldnt find bourne couldnt adjust it all went sideways finally there were no options left pamela so you had conklin killed silence i mean if were cutting the crap abbott ive given thirty years and two marriages to this agency ive shoveled shit on four continents im due to retire next year and believe me i need my pension but if you think im gonna sit here and let you dangle me with this you can go to hell marshall too flat it had to be done pamela and bourne wheres he now abbott shrugs dead in a ditch drunk in a bar in mogadishu who knows pamela i think i do we had a deal going down in berlin last week during the buy both our field agent and the seller were killed we pulled a fingerprint from a timing charge that didnt go off beat they were killed by jason bourne abbott hesitates blindsided what a courtesy knock at the door cronin appearing in the doorway theyre ready for us upstairs ',
 ' a now at the immigration officer booth bourne hands over an old blue passport it reads jason bourne whats he up to is he giving up immigration officer where you coming from mr bourne bourne tangiers the officer runs the code on the passport through the scanner ',
 ' a tech turns as a computer alarm begins an incessant beeping the screen as jason bournes passport data begins scrolling through a sleeper waking up on the grid then his photo work station as an interpol supervisor leans in over the techs shoulder to see whats up after a beat as the tech begins typing and hits send ',
 ' crewcut turns from his monitor to his own superior as at the same time ',
 ' looking up from his computer the immigration officer gestures bourne to one side immigration officer sir would you be so kind as to step over here please bourne uh sure the immigration officer comes out of his booth as a carabinieri joins him and they escort bourne to a small room at the side of the customs hall immigration officer please wait in here bourne scans the hall as he walks enters room pamelas vo seven years ago twelve million dollars was stolen from a cia account bourne takes a seat carabinieri guards the room ',
 ' same table more faces marshall back in the throne abbott three cia mandarins plus their s and pamela in warsaw this is click a photo of the man killed in berlin fills the projection screen behind her click crime scene photo of dead body click pecos oil logo pamela contd ivan mevedev senior financial manager worked for one of the new russian petroleum companies pecos oil he claimed to know where the money landed we believe this could have only happened with help from someone inside the agency this click conklins photo pamela contd placing it on the table this is conklins computer click a photocopy of a banking contract pamela contd at the time of his death conklin was sitting on a personal account in the amount of sevenhundred and sixty thousand dollars abbott do you know what his budget was pamela excuse me abbott we were throwing money at him throwing it at him and asking him to keep it dark pamela may i finish abbott conklin mightve been a nut but he wasnt a mole you have me his calendar for a couple of days ill prove he killed lincoln appealing to marshall this is supposed to be definitive pamela whats definitive is that i just lost two people in berlin abbott so whats your theory mocking her conklins reaching out from the grave to protect his good name incredulous the man is dead marshall hes heard enough no ones disputing that ward abbott for crissake marty you knew conklin does this scan i mean at all marshall signals for quiet marshall okay cut to the chase pam what are you selling pamela i think that bourne and conklin were in business that bourne is still involved more pamela contd and that whatever information i was going to buy in berlin it was big enough to make bourne come out from wherever hes been hiding to kill again to abbott hows that scan as the mandarins all start talking at once zorn enters stands at the head of the table tries to get their attention zorn hey they look up look youre not gonna believe this but jason bournes passport just came on the grid in naples abbott blinks what ',
 ' nevins american a junior cia field officer walking from the parking lot talking on his cellphone nevins what can i do i cant ill call you when i know what im into a hassled pause i dont know some guys name came up on the computer starting toward the building so start without me if i can get there i will later nevins hangs up and pockets the phone he hustles towards the building ',
 ' the room is jumping agents tracking working the phones and computers pamela giving orders abbott watches cronin looks up from computer screen looks like hes been detained pamela whos going us cronin theres only a consulate they sent a field officer out half an hour ago pamela cuts him off then get a number they need to know who theyre dealing with cronin already on it ',
 ' as nevins flashes his credentials to carabinieri at door who gives an unimpressed shrug and lets him in nevins takes his overcoat off tosses it on the empty chair we see a big ass for just a second under his suit jacket nevins alright mr bourne is that your name bourne nods names nevins im with the us consulate could i see your passport bourne silent hands over his passport nevins contd so mr bourne nevins studies bournes passport nevins contd what are you doing in tangiers silence nevins contd faux friendly are you travelling alone bourne stares straight ahead nevins comes around the table and sits in front of bourne nevins contd in his face look i dont know what youve done but youre gonna need to play ball here nevins cell starts to ring he shrugs an apology turns away and answers nevins contd contd nevins pamelaphone this is pamela landy a ci supervisor calling from langley virginia are you with a jason bourne now nevins listens looks at bourne yes ',
 ' a pamela on the phone pamela then use extreme caution he can be very unpredictable and violent use whatever means necessary to ',
 ' whatever nevins is being told its concerning bourne watching him knows exactly what this is close on nevins as he steps away listening intently his hand just starting to move toward his shoulder holster nevins contd okay ill call you right back nevins flips shut his phone he reaches for his gun even as he turns and bourne is right there in his face whump momentum and gravity reaching mutual agreement as nevins hits the deck carabinieri barely clears his holster before chop chop bourne has him down in a heap bourne is back silent and effective finding nevins cellphone bourne reaches into his bag he holds the phone next to a larger diagnostic mobile unit the confirm light blinks nevins phone has been cloned bourne puts the phone back in nevins coat takes his gun and carabinieris gun and radio and puts them in his duffle were starting to realize theres a plan at work here finally bourne exits the door wedging a desk under the handle so it cannot be opened from the inside and calmly walks away like nothing ever happened ',
 ' and now we see the old bourne in his long black coat purposely striding out of the building he pauses long enough for the security camera to get a good look at him the ronin returns ',
 ' bourne crosses the street and approaches a man putting his suitcase in the trunk of a green peugeot bourne reaches into his bag pulls out some cash ',
 ' nevins stirring the carabinieri still out a phone starts to ring nevins phone finally sitting up he answers nevins hello ',
 ' pamela at the other end of the line pamelaphone mr nevins nevinsphone whos this pamelaphone pamela landy again where do we stand ',
 ' a nevins barely knows where he is ',
 ' bourne sits in the dark car headphones a nest of cool gadgetry on the passenger seat listening in recording he writes pamela landy circles it nevinsphone i think i think he got away pamela looks at the faces waiting around the table shakes her head no pamela have you locked down the area nevinsphone ah were in italy they dont exactly lock down real quick intercut bourne nevins pamela pamelaphone how long have you worked for the agency nevinsphone me four years pamelaphone if you ever want to make it to five youre gonna listen to me real close jason bourne is armed and extremely dangerous a week ago he assassinated two men in berlin one of whom was a highlyexperienced field officer continuing as were totally on bourne at this point sitting there in the dark car struggling to make sense of this what the fuck is she talking about berlin he writes it circles it pamelaphone contd i want that area secured i want any evidence secured and i want it done now is that clear nevinsphone yes sir maam pamelaphone im getting on a plane to berlin in minutes which means you are going to call me back in and when i ask you where we stand i had better be impressed my mobile number is bourne already turning the key in the ignition the peugeot roaring to life as he writes the number dropping the car into gear bourne pulls briskly away from the curb ',
 ' a pamela finishes hangs up abbott berlin pamela ive already got a team there i doubt bournes in naples to settle down and raise a family abbott you dont know what youre getting into here pamela and you do from the moment he left treadstone he has killed and eluded every person that you sent to find him before it can come to blows marshall riot act enough i want both of you on that plane more marshall contd and we are all of us going to do what we were either too lazy or inept to do the last time around youre going to find this sonofabitch and take him down before he destroys any more of this agency beat is that definitive enough for you abbott nods sharing a look with pamela as we ',
 ' aa pamela and cronin come screaming around a corner and down a long corridor abbott and zorn trying to keep up cronin kurts reopening all the wyfi and sat links pamela uplink all relevant files to kim a look back at zorn and i want them to contact anyone who had anything to do with treadstone zorn looks to abbott as they disappear around a corner ',
 ' b the peugeot speeding north north towards germany and ',
 ' bourne driving listening to playback of pamelas conversation with nevins pamelatape jason bourne is armed and extremely dangerous bournes face eyes tight looking weird pamelatape contd contd a week ago he assassinated two men in berlin one a highly a suddenly a flashback a shard pieces lightning flash of images getting in the back seat of the car rolling brandenburg berlin a mirror the television tower the driver looks back we see him well know him later as jarda then a steel case on the backseat inside a syringe a dark vial pistol as we lay hands on them b back to b bourne out of it jolted almost losing control of the car for a second jerking back into his lane recognition toughing it out steady as she goes catching his rhythm again accelerating and ',
 ' a bakery on the corner nicky emerging nicky from the old days suddenly she stops abbott stands there beside a parked car the passenger door open message clear get the fuck in ',
 ' inside a hanger inside an office abbott watching as cronin questions nicky pamela sits on a window sill cronin so your cover at the time was what nicky that i was an american student in paris cronin what exactly did your job with treadstone in paris consist of nicky looks to abbott he nods that its okay to answer pamela bristles at the checkoff nicky i had two responsibilities one was to coordinate logistical operations the other was to monitor the health of the agents to make sure they were up to date with their medications cronin health meaning what nicky their mental health because of what theyd been through they were prone to a variety of problems pamela losing patience what kind of problems nicky depression anger compulsive behaviors they had physical symptoms headaches sensitivity to light pamela amnesia nicky before this before bourne no nicky gets agitated abbott steps in fatherly good cop abbott were you familiar with the training program nicky the details no i mean i was told it was voluntary i dont know if thats true or not but thats what i was told a bit defensive look they took vulnerable subjects okay you mix that with the right pharmacology and some serious behavior modification and i dont know i mean i guess anythings possible zorn arrives from outside zorn the jets ready points to nicky theres a car for you everybody moving nicky relieved shes off the hook she thinks she becomes aware of pamela considering her nicky good luck pamela you were his local contact you were with him the night conklin died youre coming with us ',
 ' streaks across the sky ',
 ' quiet in the cabin abbott gets up to use the bathroom pamela sits across from nicky who stares out the window as the bathroom door clicks shut pamela seizes the privacy pamela im curious about bourne your interpretation of his condition you have specific training in the identification and diagnosis of psychological conditions nicky am i a doctor no but pamela are you an expert in amnesia nicky look what do you want me to say i was there i believed him pamela believed what nicky i believed jason bourne had suffered a severe traumatic breakdown pamela so he fooled you nicky frustration building if you say so pamela leans in still low not good enough youre the person who floated this amnesia story shifts gears ever feel sorry for him for what hed been through nicky youre making it out like were friends here or something i met him alone twice pamela you felt nothing no spark two young people in paris dangerous missions life and death nicky incredulous you mean did i want a date pamela did you nicky these were killers conklin had them all jacked up they were dobermans pamela some women like dobermans nicky what do you want from me i was reassigned im out pamela see thats a problem for me nicky whatever hes doing we need to end it this isnt the kind of mess you walk away from pamela leans away nicky looks back out the window ',
 ' three in the morning as the gulf stream lurches to a stop two black sedans here for the pickup teddy the greeting party as pamela cronin abbott zorn and nicky disembark ',
 ' a the sedans making their way stopping at a nondescript office building ',
 ' b elevator opens into their th floor world emergency activity kim ready to debrief kurt work the computers energy up pamela abbott and cronin bring nicky into the room kim so far bournes had no contact with anyone on the list langley pulled an image out of naples its uploading right now kurt coming in now everything stops as the photo blurry oblique begins materializing on halfadozen monitors around the room suddenly theyre surrounded by bourne pamela to nicky is it him looking closer she nods cronin hes not hiding thats for sure zorn why naples why now pamela has gone quiet just staring at the picture as kurt could be random cronin maybe hes running abbott looks skeptical abbott on his own passport kim the image whats he actually doing cronin whats he doing hes making his first mistake and then from behind them nicky its not a mistake everyone looks over they dont make mistakes and they dont do random theres always an objective always a target beat if hes in naples on his own passport theres a reason pamela turns to abbott a silent moment between them theyre in it now and they know it ',
 ' c the peugeot streaking through the alps passing a sign for the german border moonlit glacial peaks whipping past as club music starts pulsing louder and louder and ',
 ' d bourne driving hard pushing the car through the night mission bourne as the music keeps just building and building taking us into ',
 ' packed and loud skin and smoke a doorman on the move taking us with him through the crowd faces voices all the moscow party people and at the back a vip booth kirill simply shitfaced but in a really creepy numb kind of way three women absolutely gorgeous are sitting around him chatting away as if he werent even there the girls looking up to see the doorman standing there can he walk kirill stirs his stupor a futile attempt to escape eyes still those of an exceptionally hard man a minute later kirill can walk the most graceful drunk youve ever seen making his way through the club tuning out everything but the need to get to the door and ',
 ' yes day its nine am kirill suddenly in the sunlight people going to work kids off to school and gretkov sitting in his mercedes not happy follow car and security and assistant equally unhappy gretkov you told me jason bourne was dead kirill blinking against the sunlight trying to process ',
 ' discreet and chilly a car pulls up a man gets out munich we dont see his face as he heads in ',
 ' the man enters his alarm system beep beep starts once he comes through the door theres a keypad on the wall he enters his code and the beeping stops just like everyday its a sad house he hangs his coat on the rack moving now into the kitchen he drops his briefcase on the table opens the fridge for a drink except what he comes out with is a gun wheeling around the salaryman is jarda jarda from bournes dream but as he turns bourne behind him bigger gun waiting so ready bourne i emptied it jarda a total pro felt a little light bourne drop it jarda lets the gun fall looks his old comrade over a beat but bournes not interested in a reunion bourne contd contd here bourne tosses him flexcuffs jarda puts his hands behind his back turns to let bourne cinch them bourne contd contd front use your teeth jarda caught scamming sorry old habits bourne kicks over a chair sit jarda contd word in the ether was youd lost your memory bourne checking jardas briefcase tearing through it bourne you still shouldve moved jarda i like it here a beat more jarda contd last time i saw you was greece you had a good spot bourne reacts doesnt look over but realizes jarda contd i had the girl i had her lined up that whole afternoon waiting for you that was the problem defensive you ever do two targets its tough bourne turns cold jarda contd his real question so why didnt you kill me then bourne she wouldnt let me beat shes the only reason youre alive silence jarda down a peg or two jarda what do you want bourne conklin jarda hes dead bourne the gun right to jardas face bourne try again jarda shot dead in paris dead the night you walked out bournephone then who runs treadstone jarda nobody they shut it down were the last two its over not finishing because hes falling landing hard bourne just kicked the chair out from under him bourne youre lying if its over why are they after me jarda i dont know bourne who sent you to greece jarda a voice a voice from the states someone new bourne pamela landy jarda i dont know who that is bourne whats going on in berlin jarda i dont know why would i lie silence bourne pulls back unsure jarda makes it to his feet jarda contd what the hell did you do you must have really screwed up bourne doesnt know he backs off jarda contd she really did that told you not to kill me beat i had a woman once but after a while what do you talk about i mean for us the work you cant tell them who you are bourne i did jarda hesitates its really like bourne just told him how much he loved her jarda i thought you were here to kill me something in the way he said it plus jarda just glanced at his watch bourne what did you do jarda shrugs almost embarrassed bourne looks across to the alarm pad jarda hit on the way in voltage like a switch bourne contd contd you called it in jarda im sorry bourne how long how long do i have stopping because the phone just started ringing loud insistent bourne contd contd how long ',
 ' jamming right the fuck into it three guys jarheads dod special force dudes speeding through munich jar is the driver jar is prepping weapons like a maniac in the backseat and jar on the phone its a red flag file so fix it call them back asap jar the call what whatd they do jar bad news she called munich local jar slamming home another clip its probably just a drill anyway ',
 ' phone ringing jarda in cuffs bourne scanning out the windows everything fast bourne car keys jarda my coat but we should bourne what jarda take the back get another car bourne hesitates just a moment wrong slam out of nowhere jarda swings twohands still cuffed like a mace catching bourne hard and bourne stunned jarda smashing the coffee table slices the flexcuffs through on a shard of glass free jarda follows up knee up in the ribs the gun knocked free from bournes hand skittering across the floor bourne as jarda starts to move backhanding him and ',
 ' two munich patrol cars rolling and ',
 ' seen from inside glimpsed through the glass outside its war a flatout closequarter death match jarda older and cuffed but strong and determined bourne still hammered from that opening suckerpunch the two of them braced there grappling falling jarda the cuffs hes got bourne in a chokehold but bourne driving his head back into jardas face and ',
 ' jamming along through munich ',
 ' jarda bourne the gun on the floor struggling for it jarda there first bourne on him pinned there four hands one gun and blamm wild shot into the refrigerator still wrestling breaking jardas nose until the gun knocked away again finally their hands locked into each others throats this is as real and up close as it gets until bourne finally holds dead weight eyes fixed staring bourne jumping back blood all over his shirt bournes first kill in a long time a messy one revulsion ',
 ' jarheads getting close but up ahead another munich patrol car in motion the jarheads react dont need or want the company ',
 ' bourne all business now pulling the stove away from the wall there the gas line hose bourne ripping it free gas running wide open into the room next a fork grabbing it jamming it down into the mechanism on a toaster wedging it there and now hes grabbing papers jardas stuff on the table jamming a roll of sales projections into the toaster beside the fork bourne coughing from the gas turning the toaster on checking his watch taking one last look at jarda dead on the floor and ',
 ' theyre just turning into the street ',
 ' the dod car three dods approaching the house when booooomm jardas kitchen blown out gone ',
 ' bourne same moment flying out the rear as planned urban backyard exfil hes flying and gone ',
 ' fire smoke its all burning now munich cops blown back theyll have a story to tell tonight ',
 ' drives away past arriving police ',
 ' the bullpen is cranking phones to munich lines to langley abbott watching from the sidelines kurt and kim at their work stations pamela on mobile turns to abbott pamela so he beats a man within an inch of his life strangles him then blows the place up at nicky for someone with amnesia he certainly hasnt forgotten how to kill has he across the room cronin and teddy suddenly excited about what theyre seeing on their screen cronin hey theyve got him boxed in new data coming up on the monitor everyone rushing to look excited except zorn forget it they lost him teddy whatre you talking about theyve got a three block perimeter zorn you cant see him hes not in front of you forget it hes gone cronin fuck you buzzkill its not gonna be like last time zorn you better start listening to someone cause weve been there abbott okay enough stepping in take a walk danny get some air zorn nods happy to nicky piping in i dont think we need to keep looking for him anyway pamela and why is that nicky because hes doing just what he said hed do hes coming for us and for the first time theyre all thinking the same thing ',
 ' it is pouring rain seen from that hellish car a huge distinctive needlelike tower dominates the skyline lights flashing through the dark and wet ',
 ' bournes eyes opening heart pounding springing up alone damn his side hurts recoiling from that where is he hes in the car looking around and his windshield pov an autobahn reststop gas station sleeping trucks back to bourne catching his breath shifting away from the pain in his rib checking his watch but what the hell is that on his sleeve fuck its blood jardas blood ',
 ' bourne out of the car fast careless wrong not even checking whos watching pulling off the shirt tearing it off throwing it down and standing there in the weird light a big bruise ripening on his side looking around its okay nobodys watching but shit man get it together ',
 ' a streaking along bourne back to his mission ',
 ' b roaring by a sign berlin km ',
 ' kirill striding through the terminal moving quickly toward a departure gate and gretkov above watching him go ',
 ' bourne drives up ',
 ' quiet and forlorn this early just like bourne whos taking a locker stashing a backpack prepping the evac always ready he heads outside we hear hotel operator vo front desk german berlin hilton how can i help you bournephone vo im trying to reach a guest pamela landy please hotel operator vo im sorry but im not showing that we have a guest by that name continuing as ',
 ' a bourne tucked in with a berlin guide book a felt tip pen and a fiftyeuro phonecard working it bournephone pamela landy please hotel operator sorry i dont see it here crossing out another hotel off the list four down forty to go as we start time cutting and hotel voices vo overlapping no one here by that name no sir theres no landy here how are you spelling that sir sorry but no i have no landy registered sir continuing until ',
 ' b clean and plain a bed nobodys slept in the phone begins ringing pamela fresh from the shower rushing out from the bathroom to answer it pamelaphone hello dial tone pamela hangs up that was strange ',
 ' c a taxi driving through the empty early streets and ',
 ' d bourne in the backseat staring out the window and his pov the fernsehturm looming as they pass the berlin tv tower that needle in the sky from the flashback and then e suddenly e flashback its raining were still moving still in a car still near alexanderplatz but suddenly its pouring outside turning back we realize were not in the cab anymore theres a driver up front and beside him conklin yes conklin hes in the passenger seat turning back to us handing us something a photograph a face some guy conklin neski vladimir neski the photo hes at the hotel brecker get the papers beat say it bourne treadstone bourne alone in the back staring at the photo bourne neski hotel brecker papers conklin this is not a drill soldier were clear on that this is a live project and you are go training is over bourne yes sir conklin good then gimme the damn picture back taking it see you on the other side to the driver pull over hes getting out f back to f bourne sitting in the back seat of the cab frozen there rocked whats happening to him no chance to work it out because the taxis stopped and taxi driver waiting irritated the hotel brecker or the grand make up your mind bourne what taxi driver this is the westin grand you just said brecker bourne fishing for money yeah sorry this is good ',
 ' g concentric rings looking down on each other bourne slipping in unnoticed taking a quick look up before moving along ',
 ' h bourne stepping up to the guy behind the desk the gym mostly empty bourne hi i think i left my backpack here yesterday black nike the guy disappears in back to check bourne leans across the counter scrolling the computer the guest list his finger stabbing down on screen landy pamela bourne clears the screen walks away ',
 ' j because of the setup bourne pretending to talk on a house phone has a view of room across the way the door opens pamela exits carrying an overnight bag bourne watches ',
 ' k elevator doors opening pamela coming out into the lobby heading toward the exit and ',
 ' l a black suburban at the curb cronin standing there waiting as she emerges pamela anything teddy no munichs a bust hes loose pamela are we locked up cronin i told everyone they had an hour eat sleep shave whatever they want but once were back were back for good as they pile in and bourne walking right past them hes got the whole thing scoped heading quickly across the street and ',
 ' m bourne jumps into the first cab in the rank and ',
 ' n the driver starting up the car as bourne that black suv fifty euros if you keep me close the driver smiles and ',
 ' i pt kirill walks down the same hallway gretkov came to meet him last time a guy carrying a briefcase toward him stopping for a moment to light a smoke letting kirill take charge of the briefcase smooth like it never happened ',
 ' the suv rolling up the cab continuing past and stopping at the corner ',
 ' a bourne looking back out the rear window his pov as they pile out of the van start inside acknowledged by a security detail pretending to loiter outside as we hear pamela vo munich to berlin check everything flights trains police reports thatll be box teddy thats yours continuing as ',
 ' i pt kirill opening the briefcase two automatic pistols silencers ammo care package ',
 ' a bulkhead opening bourne stepping out among the satellite dishes unpacks a bag telescope water food and we hear pamela vo box call it prior german connections nicky i want to rerun all bournes treadstone material every footstep kim box lets call it munich outbound continuing as ',
 ' weve been hearing it now were seeing it pamela at the chalkboard abbott backing her up everyone else spread around theyre regrouping urgently behind them cots are being set up food water stacked up pamela lets stay on the local cops we need a vehicle parking ticket something langleys offered to upload any satellite imaging we need so lets find a target to look for to zorn danny box i need fresh eyes review the buy where we lost the three million timeline it with what we know about bournes movements turn it upside down and see how it looks continuing as ',
 ' a decent view into the berlin hq two windows one offers a look at an empty kitchenette the other a nice shot of the bullpen area it looks like they are in for the long haul theres teddy pacing pasta glimpse of zorn conferring with abbottnow kim talking on the phone ',
 ' bourne eyes locked on the target scanning waiting and then something changes suddenly theres something down there thats clearly a great deal more electric than what hes seen so far a telescopic pov a nicky shes just come into the kitchenette pouring herself a cup of coffee nicky who he knows and bourne lowering the telescope yes now hes getting somewhere thinking it through as ',
 ' nicky is joined by pamela who goes for the coffee pamela is it fresh nicky its got caffeine in it thats all i know before pamela can pour her cell phone rings she answers pamela pamela landy bournephone i was at the westin this morning i could have killed you pamela who is this intercut with rooftop bourne its me pamela holy christ bourne nicky reacts to the name runs to the other room to try and start a trace pamela contd contd what do you want bourne i want to come in he wants to come in its like a bomb going off nicky back in with conklin pamela waving for a pencil pamela okay how do you want to do it bourne i want someone i know to take me in pamela who bourne there was a girl in paris part of the program she used to handle the medication and now we stay with pamela her eyes flicker over to nicky pamela what if we cant find her bournephone its easy shes standing right in front of you busted pamela okay jason your move bourne alexanderplatz minutes under the world clock alone give her your phone click the line goes dead pamela steps away from the window realizing hes on one of the roofs out there ',
 ' a as the bulkhead door swings in the wind bourne is gone ',
 ' b everyone gathered a big detailed map of alexanderplatz spread on the table zorn heres the clock shit hes put her in the middle of everything cronin its a nightmare well never get her covered abbott call a mayday into berlin station we need snipers dod whatever they got pamela snipers hold on he said he wants to come in abbott my ass he does youre playing with fire pamela marshall said nail him to the wall i dont know how you interpreted that but i dont think he meant repatriate him pamela dont you want answers abbott there are no answers theres either jason bourne alive or jason bourne dead and i for one would prefer the latter and what about her points to nicky you just send her out to this lunatic with no protection pamela looks to nicky pamela what do you think is he coming in nicky i dont know he was sick he wanted out i believed him pamela alright pamela gestures to abbott cronin teddy pamela contd make the call get a wire on her if it starts to go wrong take him out ',
 ' a the rear of the official berlin cia hq and here they come ten delta dudes in civvies sprinting to a couple vehicles with drivers ready and engines running and bc ',
 ' d nicky her hands overhead as zorn tapes a transmitter and battery between her shoulder blades teddy and cronin plot the area with two men plainclothed delta team kim and kurt on their own lines kim this just in they got the number bournes calls came from nevins phone the field agent in genoa teddy nevins is bourne abbott losing it are you an idiot bourne mustve cloned his phone an embarrassed silence abbott mad at himself for losing his temper looking up to find pamelas eyes on his abbott contd contd i hope you know what youre doing ef ',
 ' g in all its vastness alone theres the world clock nicky waiting on the periphery two plainclothed deltas nearby in quick succession nicky binocular pov sniper scope pov on a video monitor ',
 ' h everyone waiting holding their breath watching nicky standing as ',
 ' j nickys pamelas phone rings she answers as a yellow tram approaches bourne see that tram coming around the corner nicky yes bourne get on it she turns and walks as the tram arrives the delta dudes start moving ',
 ' k the yellow tram arrives nicky enters one of the delta dudes just barely joining her the tram begins moving nicky looks around nervously nothing happens the tram moves about yards across the platz stops at the next stop people get on and off nicky and delta dude relax a bit doors begin to close and just like that bourne swoops in beside nicky flashes a gun bourne walk bourne takes her arm and they just get off as the doors close leaving the delta dude behind they disappear down into the pedestrian subway lm ',
 ' n a madhouse a video feed on a monitor pamela wheres nicky as they realize shes gone abbott goddamn it i told you cronin listen listen he cranks the speaker bournes voice what did i say what did i tell you in paris o ',
 ' p bourne what were my words but she cant speak leave me alone leave me out of it but you couldnt do that could you nicky i didjason i swear i didi told them i told them i believed you bourne who is pamela landy nicky you hear me i believed you bourne is she running treadstone ',
 ' q pamela all ears nickys voice shes ci counterintelligence shes a deputy director bournes voice what the hell is she doing ',
 ' r nicky whats she doing nicky looks at him like hes crazy bourne why is she trying to kill me nicky they know defiant reckless they know you were here they know you killed these two guys they know you and conklin had something on the side they dont know what it is but they know as bourne tries to process ',
 ' s radio chatter going wild panic delta vo into radio where are they anyone ',
 ' t still walking bourne knowing he must be driving them nuts bourne how do they know that how can they know any of that nicky what is this a game bourne i want to hear it from you she looks at him is he crazy what bourne contd contd say it nicky last week an agency field officer went to make a buy from a russian national bourne a russian nicky it was pamela landys op the guy was going to sellout a mole or something i havent been debriefed on exactly what it was bourne last week when is she supposed to answer nicky shrugs on quicksand nicky and you got to him before we could bourne i killed him nicky you left a print there was kel that didnt go off there was a partial print they tracked it back to treadstone they know its you bourne i left a fingerprint you fucking people suddenly bournes jerking her down to a lower level ',
 ' u big static on the speakers delta co cooly checks the map delta co she must be in one of the pedestrian tunnels ',
 ' v as delta dudes fan out head for the subway entrances ',
 ' w an intersection of three tunnels bourne leads nicky far left she looks really scared ',
 ' gretkov has landed just coming off the flight a ',
 ' bourne what was landy buying what kind of files when she doesnt answer instantly what was she buying nicky conklin stuff on conklin trying not to lose it suddenly he rips the microphone out from under her shirt he knew of course dropping it as he yanks her along ',
 ' as the transmission goes dead christ aboott drills a look at pamela your fault pamela ignoring abbott that phone has a locator on it kurt and kim work their stuff ',
 ' gloomy deserted a mausoleum here come nicky and bourne she knows shes on her own now bourne dead serious looks at his watch bourne why are you here then nicky please im only here because of paris because they cant figure out what youre doing im here because of abbott bourne abbott nicky he closed down treadstone he took care of me after paris bourne so when was i here nicky what do you mean bourne for treadstone in berlin you know my file i did a job here when nicky no you never worked berlin bourne my first job nicky your first assignment was geneva bourne thats a lie nicky emphatic you never worked berlin bourne raising the gun eyes gone dead oh shit nicky contd nojasonplease bourne i was here nicky its not in the filei sweari know your fileyour first job was genevai swear to god you never worked here hes so ready to kill her nicky starting to cry hands over her face covering up bracing for the bullet she knows is coming bourne about to pull the trigger suddenly a flashback a moment a shard a womans face a backing away begging begging us begging the camera pleading for her life in russian this awful blur of desperation and panic fear too fast too panicked b jam back to b bourne swamped thrown hesitating close on nicky sobbing now when finally looking out and bourne is gone ',
 ' c an hour later whole new vibe siege mode curtains drawn three delta dudes parked around the room kurt and kim working the phones and screens the mood is dark pamela abbott cronin all in here the safe zone away from the windows cronin on a cell phone got it yeah hang on to the room okay theyve got three guys out front and another two taking the back stairs no word on nicky kurt looks up from screen even if shes still got your phone it might take awhile signals hard to trace down there pamela turns looking at the photo of bourne in naples introspective pamela so whats he doing you believe him abbott its hard to swallow beat the confusion the amnesia but he keeps on killing its more calculated than sick real soft sell what about nicky shes the last one to see bourne in paris shes the one he asks for they disappear pamela well whatever hes doing ive had enough this is now a search and destroy mission turns to the room i want the berlin police fully briefed and handing the photo to cronin get this out to all the agencies abbott agrees ',
 ' a bmw parked in the shadows ',
 ' kirill wearing headphones listening to a berlin police frequency theres an interpol wanted picture of jason bourne there on the seat hes in play ',
 ' d quiet intense activity military radios chirping here and there zorn moving through the bullpen carrying a cup of coffee heading back toward pamelas office where abbott is leaning in the doorway past him inside we can see pamela in the midst of a tough phone conversation cronin and the delta boss sitting there with her zorn the coffee sir abbott thanks abbott nods takes a sip looking beat zorn contd i have that number you wanted abbott hesitates but only a moment he never asked for a number but hes playing along looking satisfied as zorn hands him a slip of paper abbott glancing at it she say what time i should call zorn the sooner the better abbott nods pockets the paper turning back as if it were nothing and ',
 ' e massive modern busy bourne in the back in a corner doing a search hotel brecker scrolling and then stopping freezing because on the monitor a berlin newspaper archive there it is written large in loud tabloid german oil reformer murdered theres a photograph of the berlin police carrying two body bags out of the hotel brecker theres a caption identifying the dead as vladimir and sonya neski theres even a long article accompanying all this but its in german and we dont need to read it anyway because bourne is reading it and were reading in his face that he is rocked that he has found another bottom to the abyss ',
 ' f remember the building where vic was killed were back zorn and abbott making their way in zorn steering them away toward a stairwell at the back ',
 ' zorn and abbott have snuck in here work light signs of repair on the wall zorn nervous i did my box work but i wanted to show you before i showed landy i came out here last night because none of this was making any sense i mean im with you on this conklin was a nut but a traitor i just cant get there abbott what do you have danny zorn the electrical riser you put a fourgam kel on here and its gonna take out power to the building you know that what you cant know is if its gonna blow the room with it abbott and zorn there were two charges they were supposed to go off simultaneously the second one the one that didnt go off was down here pointing it out first of all this is nothing its a sub line for the breaker above second why put the charge all the way down here if youre good enough to get in here and handle the gear youre good enough to know you dont need this beat bourne would know abbott it was staged zorn is it a slam dunk no but abbott jesus zorn spitballing okay what if someone decided to cover their tracks by blaming conklin and bourne what if bourne didnt have anything to do with this abbott keep going zorn somethings been going on here in europe and its still going on post conklin whos been in berlin abbott lots of people zorn including landy jumping off the cliff she had access to the archives zorn hesitates but its out its in the room abbott who else knows about this zorn nobody you hes scared i had to tell you right abbott show me again zorn okay turning away when abbott out of nowhere his hand jamming up into zorns ribcage more than his hand because zorns eyes barely have a moment to register shock before they bulge clenching the younger mans body pulling him close as he turns the knife and zorn is dead abbott without hesitation shifting away from the blood letting the body fall abbott standing there listening checking himself for blood hes clean looking for a place to stash the body as ',
 ' a bourne across the street staring at the hotel haunted as a police siren edges closer through the empty streets aa flashback aa we are a pov a stakeout watching the hotel across the way the pov checks its watch checks the perimeter the street deserted foreboding the hotel our destiny waiting up there somehow and suddenly a light comes on a terrible signal and as the car suddenly lurches forward and around the corner ab back to ab bourne muscling up his backpack heading toward the hotel ',
 ' b and hotel fusty but comfortable and busy guests and staff doing their thing a clerk behind the reception desk clerk guten abend bourne playing it american guten abend clerk switching to english can i help you suddenly ba flashback the lobby but seven years ago ba across the room a man buttoning a raincoat as he passes neski bb jamming back to bb bourne stalled coming back as clerk contd contd sir smiling do you have a reservation bourne no sorry i just got in rallying back i is room available off the clerks look i stayed there before my wife and i the clerk nods checking the register the concierge just down the desk glancing over at bourne nodding hello and clerk im sorry that room is occupied would room be okay its just across the hall bourne sure thats fine danka cd shot ',
 ' a bourne riding up alone dread mounting and ',
 ' the concierge coming out of the office with a sheet of fax paper placing it quietly down beside the clerk and the fax bournes face the same wanted picture and ',
 ' bourne off the elevator he makes his way down his pov the sixth floor hallway suddenly scary ',
 ' a kirill sitting up as the police radio starts broadcoasting an allpoints bulletin the words hotel brecker in there kirill dropping the car into gear and ',
 ' b bourne walking theres his room but across the hall and down one room bourne steps up listening a moment then he knocks nothing he pulls a knife from his pocket checks the hallway hes clear wedges the blade in there and onetwo pop ',
 ' bourne enters a suite closing the door behind him and treadstone bourne seven years ago does the same bourne shakes off the flash looks around the lights are on an open suitcase on the bed ',
 ' the clerk the concierge and the manager are huddled in conversation with three berlin cops whove just arrived and trying to be discreet but this is clearly serious ',
 ' bourne just standing there breathing it in treadstone bourne doing the same ',
 ' bourne with his hand on the wall as if he can feel it like its all still here heart pounding and ',
 ' chaos bournes been found everybody rushing out cronin to teddy go take the van pamela the hotel how far teddy five six minutes cronin kurt youre here keep the comm line open ',
 ' bourne standing there looking out the window the images the television tower over the city everything but the rain ',
 ' the berlin police swat team truck arrives discreetly by the back loading area ',
 ' bourne flat against the wall just as he was leaning forward to see in the mirror just so and there ',
 ' a a man in the mirror pacing into view neski on the phone a talking in russian its raining bourne standing there treadstone bourne still wet from the rain one eye on that mirror and the other on a syringe that he prepped a predator the mirror the doorbell rings neski gets off the phone bourne tensing new element factoring and the mirror as neski opens the door a new flood of russian happy its mrs neski a surprise but hes very happy to see her bourne pocketing the syringe new weapon pistol quiet methodical watching the lovers bill and coo and the mirror mr neski kisses her takes her bag shes hanging up her coat and moving now toward the bathroom and bourne checking the window the weapon his balance and the mirror mrs neskis face right there seeing him so freaked she cant even register it yet bourne with the pistol in her face finger to his lips shhh but she knows backing away begging for her life in russian this awful blur of desperation and fear mr neski turning back to see his wife backing out of the bathroom and bourne with the pistol with no hesitation snap one shot into neskis heart hes down mrs neski whats just happened bourne has her wrist in his hand raising it to her head to where he holds the pistol her fingers his trigger snap letting the gun fall with her as she drops and bourne starts to move starts to prep his evac but theres something on the dresser a photograph the neski family father mother and a twelveyearold girl arms around each other happy and bourne staring at the picture undone for a moment hard out flashback to ',
 ' bourne our bourne standing where they fell frozen there paralyzed by the shame of original sin pt ',
 ' a swat captain conferring discreetly with the manager manager hes in swat captain call all the guests on the th floor tell them to remain in their rooms tell them its a police order then start on the th and th floors ',
 ' a bourne trying to stabilize to breathe ',
 ' the swat team on their way up ',
 ' a ring ring bourne snaps back as the phone in his room starts to ring four times and it stops bourne freezes footsteps shadows under the door he leans into the peephole bournes pov room german swat team taking position ',
 ' b bourne backs away surveys the room his watch his balance and ',
 ' c quickly turning into a major event halfadozen police vehicles already parked here more arriving every minute passersby mixing with the cops and people from the hotel whove just come out and kirill jogging over from the bmw hes just parked and ',
 ' wham the door kicked off its hinges swat team flooding into bournes empty hotel room and ',
 ' a bourne in motion out the bathroom window and ',
 ' berlin swat leader gives order to search other rooms and ',
 ' bourne up the water pipe to the roof as he arrives a swat team member turns bourne pulls him over the edge fires point blank into the nd swat members vest stunning him hes moving fast scrambling along the roof and into the night ',
 ' wham the door caves in and the swat team moves enters rushing to the window nobody no sign of him and ',
 ' kirill heading for the hotel entrance blocked by the exiting guests ',
 ' too many cops and radios swat team boss trying to take charge listen up were clearing the building room by room ',
 ' pamela jumping out of a van the moment it stops seeing it all the crowd the army of cops the searchlights playing across the hotel facade its another disaster ',
 ' kirill wants to get upstairs he cant too many guests coming down the stairwell berlin cops trying keep it moving and ',
 ' kirill hears bourne is on the roof ',
 ' pamela and cronin listening to teddy who just got the police update teddy black coat possibly leather dark slacks dark tshirt pointing now he says theyre gonna try and corral the guests on the street over there and then check them out but pamela disgusted yeah thatll workwhat the hell was he doing here cronin maybe he just needed a place to spend the night pamela i want to look at the room to teddy as she goes check it out pamelas in charge now they enter the elevator ',
 ' bourne coming around the other side of the hotel stepping to the left before he spots the swat van bourne aboutfaces heads the other way a sidewalk cop looks over checks the bourne photo print out in his hand ',
 ' teddy huddled with the hotel manager and a group of high ranking berlin cops turning back as abbott arriving breathless they missed him teddy so far but they found nicky shes back at the westin bourne let her go abbott he let her go great wheres danny he should head over there and debrief her the hotel whats here what was he doing teddy we dont know theyre in a room upstairs i was told to wait down here abbott accepting that because he has to only we see the fear turns to leave abbott ok if you see danny tell him i went back to the hotel abbott steps out into the street as ',
 ' bourne striding away and following sidewalk cop blowing a whistle fumbling for his holster bourne running now slowly at first and ',
 ' a now faster as if he can gauge his speed and distance ',
 ' motion bourne tearing away and ',
 ' a bourne slows to a walk two patrol cars heading his way no choice there a narrow passageway between two moving trolley trains and sprinting through the patrol cars skidding into s ',
 ' b the river spree lit by the trolley thats rumbling past and the running lights of a double coal barge up the river bourne runs across the bridge going as fast as he can hearing the police sirens swirling behind him when a third and fourth police car ahead bourne turns hard for a stairwell jumps the walkway curb leaps up the stairs two at a time as all four cop cars skid to a stop as doors open ',
 ' a tram waiting as the last few passengers get on the doors seem to stay open in slow motion as bourne appears makes a mad last dash and hes on and the doors dont close its not scheduled to go yet and here come the cops bourne off the tram guns appear bourne runs to his left stops short the other cops are coming this way screaming at him not a lot of options bourne looks over the rail down below a coal barge passing the prow just emerging bourne on the rail and jumping even as the first shot is fired ',
 ' bourne lands hard stands voltage going up one leg and theyre shooting at him he can worry about the leg later he runs back toward them the barge moving slow bourne disappears under the bridge ',
 ' guns aimed police waiting for a clear shot two of them dash to watch over the other side ',
 ' countering the barge going one way bourne the other dodging all the superstructure on deck all the while keeping his cover overhead and leaping to the second barge and more of the same until bourne running out of barge leaping back onto the bridge footing and ',
 ' the police watching the barge fully emerge continuing down river shouting in german that hes either in the water or hiding on the barge off they go down the stairs leaving the passengers on the tram blinking out in shock and bourne climbing back over the rail limping back on the tram just before the doors close and off it goes ',
 ' police converge from both ends barge goes under as kirill arrives at the center of the bridge missed again behind kirill a train snakes off into the night ',
 ' pt pamela and cronin move into the living room a couple of cops in the hallway outside cronin the room he checked into was across the hall why why would he come here pamela glances around something bothering her about this space pamela he mustve had a reason thats how they were trained cronin moves around the bedroom then into the bathroom and cronin he went out the window in here ',
 ' pt there on the mirror scrawled in soap on the glass i killed neski cronin pam you need to see this pamela moves in behind him cronin contd whos neski both of them staring pamela thinking alrighttake it down cronin what pamela this stays between you and i sensing confusion we finally have an edge i dont want to lose it ',
 ' very late abbott waits on an isolated bridge a lone figure in the shadow of east berlin gretkov arrives by car walks through the darkness abbott barely glancing over abbott you told me bourne was dead gretkov there was a mistake abbott ill say you killed his goddam girlfriend instead now theyre onto neski theyre at the brecker hotel even as we speak gretkov will it track back to us abbott no the files are spotless whatever they find its just going to make conklin look worse gretkov and the landy woman abbott shes done everything i wanted she bit on conklin so fast it was laughable she even found his bogus swiss account gretkov anything else abbott shoves a piece of paper and address into gretkovs hand abbott the paper theres a body in the basement danny zorn hes got to disappear for good clean and fast ill put him in bed with conklin and bourne even the girl nicky give me twentyfour hours ill think it up but get the goddamn body out of there its getting late a taxi now and then abbott contd neski was a roadblock without me theres no company no fortune you owe me uri one last push gretkov one last push one gretkov leaves abbott watches him go ',
 ' seconds later gretkov getting in slowly ',
 ' kirill slouched in back waiting gretkov to the driver gretkov airport to kirill were done here kirill nods as they pull away abbott turns and walks into the foggy night ',
 ' a late abbott walks a lonely figure past someone in the shadows bourne mr abbott he turns to answer when bourne firmly guides him into a side street bourneabbott scene ',
 ' as pamela and cronin exit the elevator they are met by teddy teddy heres what ive got reads remember vladimir neski russian politician seven years ago he was due to speak to a group of european oil ministers here at the hotel he never did he was murdered pamela by who teddy his wife in room then she shot herself pamela and cronin share a look pamela to teddy alrighti want you kurt and kim to stay on bourne track everything thats out there teddy goes to get in the van pamela follows with cronin pamela contd confidentially to cronin and i want you to go through and cross reference our buy that went bad the neskis and treadstone as they get in pamela contd they have to be related ',
 ' bournes arrived limping as he continues for the station ',
 ' bourne retrieving the exfil bag he stashed in the locker changed his clothes ',
 ' bag slung limping out bourne has changed clothes a big overcoat knit cap ',
 ' a busy midnight departure big train bourne climbing on the train under the sign moscow express moved ',
 ' a a blueprint spread across a table nicky kurt kim all gathered around cronin works the treadstone files on another table teddy at center briefing pamela teddy were looking at all berlin outbound good news is every train station in berlin has thirty to forty fixed digital security cameras common feed pamela are we hacking or asking teddy yes in that order pamela and what about you anything cronin its starting to link up the hijacked money the leak pecos oil one last bit is treadstone ',
 ' crossing the border into poland cold desolate snow ',
 ' conductors moving quietly through the dark cars checking tickets and visas and bourne hands over his ticket and russian passport off the grid ',
 ' a am kurt kim and teddy spread around the room theyve been running laptop train station videos for hours just about ready to raise the white flag all they have so far is an isolated loop of bourne limping into the mens room cronin watches it stutter along cronin does it look like hes faking teddy on the way in forget it kurt the legs definitely hurt cronin the blueprint well theres no window in the mens room folks so lets find somebody coming out with a bad left leg kurt worn out maybe hes still in there teddy ive got a limping guy but its the right leg kim walking away or walking toward you cronin jumping on that right there over teddys shoulder cronin thats him its the coat what train is that ',
 ' bourne asleep in his chair rocked by the rhythm but something wakes him up looks out the window something weird about the light out there then up to see marie looking at him over the back of his chair in front of him no big deal bourne hey she smiles a beat she comes around sits beside him he looks away out the window bourne contd i wanted to kill him marie but you found another choice bourne i did marie it wouldnt have changed the way you feel bourne it might have bourne looks back at her she smiles he accepts it leans back closes his eyes bourne contd i know its a dream marie you do bourne i only dream about people who are dead marie leans over kisses his forehead whispers bourne contd god i miss you i dont know what to do without you marie softly serenely jason you know exactly what to do that is your mission now bourne opens his eyes and its morning outside and marie is gone a little girl smiles at him from over the back of the chair in front bourne cant meet her gaze for long as he looks back out the window ',
 ' bourne watching the birch trees rush past not quite hiding the smokestacks beyond eyes locked forging something within one final mission as we ',
 ' abbott coming through its empty this early but heres pamela nicky cronin and the team waiting to report pamela sorry to wake you abbott waves off apology i wasnt sleeping to nicky as he passes you ok nicky yeah thanks abbott whats up pamela bunch of stuff pamela looks to cronin him first cronin we tied the room bourne visited tonight to a murdersuicide seven years ago a russian couple the neskis abbott playing along neski the reformer i remember that cronin he championed the equal distribution of oil leases in the caspian sea when he died they were all released to one petroleum company pecos oil guess what the ceo uri gretkov is ex kgb nicky someone was using treadstone as a private cleaning service abbott conklin a beat its im sorry pamela i guess you were right all along pamela waves him off its okay but pamela theres something else abbott can see by their faces this hits closer to home abbott what pamela they found danny zorns body dead in the basement at the building where my people got hit the first time abbott oh god it must have been bourne pamela did he say anything to you abbott no it must have been bourne pamela straight pamela well know for sure when we get the security tapes cronin but we can relax we tracked him hes on a train to moscow abbott reeling hiding it abbott moscow what the hells he going to moscow for pamela shrugs dont know abbott jesus i zorn i have to call his family tell them pamela im sorry ward they watch as he goes ',
 ' abbott in the rising elevator imploding ',
 ' palatial but you cant buy taste gretkov working his computer answers his phone gretkov da abbottphone you didnt stay uri gretkov matter of fact this is not a clean phone ',
 ' everyone still here cronin answering his cell phone motioning to them hes got news cronin phone to his ear youre sure pamela what the tapes cronin nodding but hold on holding the phone yep and abbott just direct dialed moscow from his room now we realize shes set a trap and abbotts walked in all the same pamela shakes her head wishes it wasnt true and theyre moving ',
 ' abbott at his desk still on the phone pouring a vodka gretkov leaving was a business decision were both rich come enjoy it abbott what do you mean gretkov go to the airport get a plane ill have a brass band waiting for you abbott save it for bourne gretkov what theres a knocking at his door abbott simply ignores it abbott he left yesterday on the night train hes probably just getting in now he drinks youll have to hurry gretkov bourne comes here why more knocking abbott good luck ',
 ' a speeding east through the russian countryside the forest is gone replaced by factories and refineries a wasteland of rust and gray that seems to go on forever ',
 ' pamela knocking again nicky teddy and cronin behind her pamela open it cronin with a pass key teddy prepped and ',
 ' a pamela leading they enter stop short abbott at his desk calmly pointing a pistol at pamela abbott they go you stay she looks back cronin shakes his head no pamela yes now they reluctantly obey the door clicking shut behind them abbott sit down pamela id rather stand if its all the same to you abbott i dont exactly know what to say im sorry pamela why would be enough for me abbott im not a traitor ive served my country pamela and pocketed a fair amount of change while doing it abbott why not it was just money pamela and danny zorn what was that abbott had to be done pamela no good options left abbott shrugs in the end honestly its hubris simple hubris you reach a point in this game when the only satisfaction left is to see how clever you are pamela no you lost your way abbott well youre probably right i guess thats all that hubris is he raises the gun pamela presses her lips together closes her eyes boom she opens them and as cronin flies back through the door theres abbott dead at the desk hes shot himself also in a way with some help from bourne ',
 ' the train easing to a stop the platform busy with people waiting and passengers disembarking bourne among them unremarkable in the crowd and ',
 ' bourne on the move welcome to the whole mad moscow scene a jumble of faces and voices travellers arrivals and departures families beggars drunk war vets hawkers ',
 ' there in the plaza bourne hobbling across the street when suddenly a car horn he turns and look out a big black bmw speeding past followed by two more all three cars with blue lights strobing on the dashboards a convoy whipping by like they own the place and taxi driver os gangster bastards dont care what they do bourne turns a grizzled taxi driver right beside him bourne pulls a slip of paper from his pocket bourne his russian is basic you know this address the taxi driver squints finally grunts affirmative he motions to his cab as they get in and pull away ',
 ' lots of cars no people but someone running its kirill pulling his keys as he sprints past and ',
 ' bourne and the taxi driver looking over as three moscow police cars speed by sirens wailing taxi driver its always something right bourne just nods as we ',
 ' kirill at the wheel a guy in a hurry who knows what hes doing one more thing on the passenger seat two big automatic pistols ',
 ' moscow cops fanning through the crowd showing bournes interpol picture have you seen him ',
 ' moscow cops with the picture flashing it around until young cabby the moment he sees it he was just here they just left ',
 ' theyve stopped bourne flashes a fifty dollar bill bourne you wait you understand stay taxi driver happy to pocket the cash sure no problem i sit ',
 ' old moscow but not for long theres new construction metastasizing all around it bourne crosses the street and his pov an abandoned wooden house windows shattered and boarded up paint all but gone roof and gables all failing back to bourne crestfallen checking the address this is it ',
 ' more cops everything focused on another taxi driver whos making a call on a cell phone everybody waiting on it ',
 ' bourne off the sidewalk now peering around the side trying to see if theres anything around back and over there an old woman on the steps next door watching him bourne starts over finding the sweetest smile hes got ',
 ' the taxi driver still parked there his pov bourne and the old lady shes pointing like shes giving directions when suddenly the drivers cell phone rings taxi driverphone hello ',
 ' bourne and the old lady his russian is limited but shes charmed nonetheless bourne a pento writeone minute searching his pockets ',
 ' the taxi driver on the phone not so happy anymore taxi driver im looking at him american hes right here ',
 ' the old lady scribbling on a piece of paper bourne reacting as the taxi drops into gear pulls away bourne wait hey but the taxi only speeds up and ',
 ' moscow police cars tearing away and ',
 ' kirill driving reaching for his ringing phone and ',
 ' the black bmw a moment later slamming on the brakes fishtailing a uturn and ',
 ' bourne hustling past all the new construction glancing back as police sirens start rising behind him and ',
 ' kirill skidding around another corner and ',
 ' two police cars just stopped there cops the old lady pointing everyone turning as the red lexus speeds past them and ',
 ' bourne coming down as fast as he can just ahead theres a footpath beneath a four lane overpass a neighborhood on the other side he could disappear there ',
 ' kirill driving and scanning there as he passes it the overpass slamming on the brakes and ',
 ' bourne hobbling out in the open twenty yards to go ',
 ' kirill jumping out of the lexus with a pistol in hand and ',
 ' bourne no clue bang his shoulder hes hit he throws himself forward and ',
 ' kirill shifting for a better second shot and ',
 ' bourne hes diving rolling pure instinct back under the embankment and ',
 ' kirill with no shot suddenly leaning over the rail just as the two moscow police cars come screaming up moscow cops jumping out with guns drawn and ',
 ' bourne hes up hes bleeding hes moving and ',
 ' chaos kirill with his hands in the air moscow cops coming toward him everyone screaming moscow cops mockbourne up hands up keep im kgb assholes them up drop the gun were chasing the same guy drop it hes getting away they let kirill go he looks back at the footpath bourne is gone as ',
 ' a gretkov strolls along suddenly two black sedans pull up and he is arrested ',
 ' a bourne hurriedly makes his way to the other end a few beats later kirill on the hunt ',
 ' a labyrinth of stalls food hardware clothes and crowded even this hardtoimpress crowd noticing bourne hobbling through nothing like a limping madman with a fresh gunshot wound to get attention people back off pull their kids out of the way some woman starts screaming and ',
 ' a security guard hears the commotion jogs out and ',
 ' kirill running toward the market five moscow cops behind him cant keep up and ',
 ' the security guard coming up fast behind bourne security guard hey hey you stop bourne turns the security guard right behind him and bourne no warning his good arm smash right into the security guards face and bourne takes his pistol and the crowd they jump holy shit ',
 ' crazy kirill sprinting through where did bourne go ',
 ' bourne back on the march except now hes shopping grabbing a bundle of tube socks and ',
 ' kirill sprinting out toward the stalls and ',
 ' bourne there a roll of duct tape and a bottle of vodka and ',
 ' kirill fighting his way through the fleeing crowd ',
 ' pt bourne leaving the market taking a swig of vodka and continues knows there are two new cops on his ass ',
 ' pt another cab stand cabbie by a yellow cab looks up to see bourne coming toward him and also the two cops as bourne nears the cabbie shakes his head bourne pivots casually like he doesnt know theyre coming until he spits vodka into one of the cops face blinded as bourne takes him and his partner out the cabbie raises his hands in surrender steps aside as bourne takes his car ',
 ' pt bourne in the yellow cab starting the engine peeling away careening into the street and kirill sprinting into the parking lot just in time to see ',
 ' pt bourne concentrating away the pain trying to drive ',
 ' two ladies ducked behind a big black gwagon freaked out as kirill grabs their keys and ',
 ' the cab speeding across a boulevard into an older neighborhood of rising narrow streets and two moscow police cars pulling uturns on the boulevard whipping around to give chase and the gwagon in full pursuit now and bourne driving up this curving little hill and the two moscow police cars starting to climb and kirill driving and hes on the hill now bourne bad hand on the wheel holding on trying to find something in passenger seat tube socks the two moscow police cars splitting up one on bournes ass the other cutting hard into a side street flanking him and bourne topping the hill two choices right or left right no wrong because down the hill theres a police car just about to angle in from the sidestreet and bourne no choice flooring it the cab its a whale slam knifing the front end of the police car and the police car spun back crashing against a building on the corner and kirill right behind that guy swerving onto the sidewalk sparks from the wall as he scrapes hanging in skidding into a turn down the hill and just missing the first police car bombing right past him bourne in pain as he packs his shoulder wound with the socks ahead the street banks downhill to left and there a boulevard wide ride lots of traffic and the cab rocketing into the flow and behind him police car with the gwagon right on his ass and bourne wrists flicking the wheel the cab screaming through the slower traffic and kirill totally on it pedal down passenger window open wind blowing hes got the pistol in his hand closing the gap and the black gwagon blowing past police car and bourne steering barely as he tears a few strips of duct tape to finish his triage blam blam the gwagon right beside him bourne reacting what the fuck thats not a cop but no time to clock kirill because kirill shit cant keep shooting into the oncoming lanes swinging wide a truck swerving again and the cab wavering again rallying and up ahead the boulevard opens into the river beltway big wide fast kremlin in the bg and four new police cars screaming down from red square and bourne skidding onto the beltway looking for room finding it open road kirill back in the hunt and the river beltway cab screaming past then one two three four police cars now the black gwagon and bourne both hands on the wheel hes already forgotten about his shoulder the beltway up ahead another choice right takes you up to the city left is a transit tunnel and bourne checking his rearview starting right and the two lead police cars right on his ass and bourne fake out veering left last second into the tunnel and the two lead police cars wrong and worse trying to change crash spinning and its not just them a third police car caught in the clutter not to mention the commuters crash the police are out of the race kirill not fooled threading the needle through the carnage and into ',
 ' four lanes two way and long theres the cab squibbing past slower cars and kirill on him move for move follow the leader and bourne checks the rearview hes lost them all but the gwagon who the hell is that the heavyweights world championship belt up for grabs kirill gaining nearly pulling level bourne nowhere to go thats never stopped him before he carves a path turns two lanes into three as sparks his way through a lane split the gwagon roaring after him bourne checks the mirror closer who the hell is that guy kirill gaining firing through his passenger window bourne brakes tunnel as the two vehicles scrape along each other kirill firing back odd angle bourne ducking for meager cover as bullets stitch through the roof tunnel the gwagon crushes the cab against the wall sparks showering the windshield finally the cab shoots ahead kirill in a controlled fury the suv jerking hard and right into the rear of the cab bourne trying to keep control spots a maintenance truck up ahead kirill banging away as his quarry straightens maintenance truck looming bourne a hard left tunnel the cab wrapping around the front of the suv wham pushing it to the right the cab continues spinning around the gwagon details front bumpers locking on rear fenders as tunnel the gwagon hurtling forward the cab ass end first locked together kirill firing into the cab really unloading now bourne down on the floor a tornado overhead kirill slaps in a new clip intense bourne gun against his door just below the window knob whumpwhumpwhump suv tire shredding kirill fights the wheel another truck looming large bourne looking between the seats out the rear window a lane dividing pillar ahead cab as bourne sits up jerks the wheel to the right tunnel the cars unlock spin away from each other kirill focused taking deadly aim bourne staring back at him calm i know something you dont know kirill frowns the truck swerves to reveal the pillar to kirills pov kirill eyes go wide whallop steel vs concrete concrete victorious a bone compressing truly horrendous impact bourne whipping the wheel cab spinning to a stop out of harms way door opening ',
 ' gun ready bourne heads over ahead spam in a can bourne crouches down looks in kirill bloody beattocrap barely alive but trapped entombed alive by the metal crushed around him bourne watches not here to help kirill looks over calms a moment as the two men consider each other bourne looks at him long and hard kirill dies and bourne stands and just walks away ',
 ' a snow swirls pamela disembarks from the g or us military plane she is met by russian officials ',
 ' huge awful sovietera housing towers fill the horizon a city bus grinds to a stop people trundle off working people at the end of their day tired cold a girl trudging a manmade wasteland twenty a proud little waif sad eyes home from some job irena ',
 ' grimmer up close rusted steel mesh over the windows drunk teenagers a haze of cigarette smoke irena pushing through doesnt want to talk to anyone ',
 ' irena climbing a junkie here flickering light there ',
 ' irena her key at the door domestic disturbance playing across the hall she opens up and ',
 ' its dark and shes barely through the door when irena jumps chokes back a cry bourne is standing there propped there actually behind her gun in hand motioning for her to be quiet bourne his shabby russian quiet silence okay irena nods scared gun in hand bourne pushes the door the last few inches so its fully closed irena i have no money no drugs is that what you want and now she can really see him hes a disaster shivering bloody eyes more hollow than hers are bourne sit can you trying to conjure the russian the chair have the chair irena accented i speak english bourne staring at her nods gestures for her to sit bourne please so she does and here they are bourne contd contd of all the people in the world youre the only one i have anything to offer hesitating thats why i came here irena shes terrified okay hes got something beside him something hes taken off the wall its the photograph the neski family same as the one that was in the hotel brecker mom dad and irena arms around each other in front of the house before it was abandoned happy smiling perfect bourne its nice a beat does this picture mean anything to you no answer hmm irena its nothing its just a picture bourne no its because you dont know how they died irena he couldnt understand no i do a change in bourne as he studies her measures her some moment of truth is here irena braces unsure bourne i would want to know beat i would want to know that my mother didnt kill my father i would want to know that she didnt kill herself irena what she really looks at him now fear overwhelmed by curiosity bourne i would grow up thinking that they didnt love me if they just left me like that irena making sure her eyes dont leave his they dont bourne contd contd it changes things that knowledge doesnt it irena wary yes bourne thats not what happened to your parents irena then what bourne i killed them body blows but he has her attention she wipes a tear bourne contd it was my job my first time your father was supposed to be alone but then your mother she came out of nowhere a little shrug i had to change my plan beat you understand me does she you dont have to live like that anymore thinking that irena you killed them bourne nods thats right bourne they loved you beat and i killed them irena howhow canhow can you be here and say this bourne i dont want you to forgive me she stands suddenly stands because if she doesnt shell burst into tears because she knows if she starts crying she wont be able to make sense of this irena for who he doesnt answer killed for who bourne pushes himself to his feet a real effort bourne it doesnt matter your life is hard enough irena youre a liar bourne you know im not irena youre a liar bourne look at me there they are two people standing in a room squared off and now she starts crying really crying and hes taking it irena i should kill youif its true you should diei should kill you now bourne i cant let you do that either irena because youre afraid bourne no starting for the door because you dont want to know how it feels she hesitates stunned hes leaving hes opening the door bourne contd i have to go now irena is this really happening bourne empty im sorry and she sags back into the chair as the photograph on the table the sound of the door closing and irena crying as ',
 ' bourne trudging along across the snow hes done it and he really cant take another step theres a bench he sits down out of gas he just might die here we slowly tilt up to the multi colored moscow tenements fade out ',
 ' bourne waking up sitting up where is he trying to get his bearings but its so bright white walls sheets sunshine through clean windows and pamela os hello david there she is standing at the foot of his bed bourne where am i pamela ramstein air base germany smiles before the wall fell you would have woken up in a russian prison hospital he looks around tries to move hammered by pain bourne oh shit pamela careful long moment hes taking it in trying to bourne why am i alive pamela are you disappointed they study each other a beat bourne i know who you are pamela nods very calm here no sudden movements pamela thank you for your gift im sorry about marie bourne whats that pamela do you think you can read are you well enough she has a folder a photograph bournes face stapled to the cover pamela contd its all in here treadstone a summary of your life all of it he waves it off bourne dont need it i remember everything pamela smiles again sounds like a threat bourne you didnt answer my question pamela why youre alive beat youre alive because youre special because she kept you alive she smiles because we want you back on our side bourne silent but hearing it pamela leaves the file pamela contd contd take a look at it well talk later bourne watching her back away as she exits into ',
 ' long sterile hallway cronin and nicky standing there with an air force sentry assigned to guard the room cronin and nicky trying to play it cool but now as they get some distance down the hallway pamela to the sentry lets give him half an hour nicky quietly so pamela felt promising its a start a chill in the air both of them going quiet because theres a nurse carrying a tray of food shes coming toward us theyre walking away staying with the nurse now coming up the hall the sentry smiles opens the door and she enters ',
 ' empty bed open window bourne is gone as the music starts pumping and we ',
 ' off he goes disappearing into thin air fade out the end ']
</pre>
##### Create BoW Vector



```python
from sklearn.feature_extraction.text import CountVectorizer

# filter stop words
vect = CountVectorizer(tokenizer=None, stop_words="english", analyzer='word').fit(corpus)
bow_vect = vect.fit_transform(corpus)
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)
```


```python
word_list[:5]
```

<pre>
['aa', 'ab', 'abandoned', 'abandons', 'abbott']
</pre>

```python
count_list
```

<pre>
array([ 3,  3,  2, ...,  1, 42,  3], dtype=int64)
</pre>

```python
bow_vect.shape
```

<pre>
(320, 2850)
</pre>

```python
bow_vect.toarray()
```

<pre>
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
</pre>

```python
bow_vect.toarray().sum(axis = 0)
```

<pre>
array([ 3,  3,  2, ...,  1, 42,  3], dtype=int64)
</pre>

```python
word_count_dict = dict(zip(word_list, count_list))
word_count_dict
```

<pre>
{'aa': 3,
 'ab': 3,
 'abandoned': 2,
 'abandons': 1,
 'abbott': 128,
 'abbottnow': 1,
 'abbottphone': 4,
 'abbotts': 3,
 'abend': 2,
 'able': 1,
 'aboott': 1,
 'aboutfaces': 1,
 'absolutely': 1,
 'abyss': 1,
 'accelerating': 1,
 'accented': 1,
 'accepting': 1,
 'accepts': 1,
 'access': 3,
 'accompanying': 1,
 'accomplished': 1,
 'account': 3,
 'acknowledged': 1,
 'act': 1,
 'activity': 2,
 'actually': 3,
 'address': 3,
 'adjust': 1,
 'adrenaline': 1,
 'affirmative': 1,
 'afford': 1,
 'afraid': 1,
 'afternoon': 1,
 'againi': 1,
 'agencies': 1,
 'agency': 5,
 'agent': 2,
 'agents': 2,
 'agitated': 1,
 'ago': 10,
 'agreement': 2,
 'agrees': 1,
 'ah': 1,
 'ahead': 17,
 'aim': 1,
 'aimed': 1,
 'air': 9,
 'airport': 2,
 'alarm': 3,
 'alert': 1,
 'alexanderplatz': 3,
 'alive': 9,
 'alley': 2,
 'alleys': 1,
 'allocation': 1,
 'allpoints': 1,
 'alongside': 1,
 'alps': 1,
 'alright': 2,
 'alrighti': 1,
 'alrighttake': 1,
 'american': 4,
 'ammo': 1,
 'amnesia': 5,
 'amused': 1,
 'anger': 1,
 'angle': 2,
 'ankle': 1,
 'anonymous': 3,
 'answer': 8,
 'answering': 1,
 'answers': 7,
 'anymore': 3,
 'anythings': 1,
 'apology': 2,
 'appealing': 1,
 'appear': 1,
 'appearing': 1,
 'appears': 1,
 'approached': 1,
 'approaches': 2,
 'approaching': 1,
 'archive': 1,
 'archives': 2,
 'area': 5,
 'arm': 2,
 'armed': 2,
 'arms': 3,
 'army': 1,
 'arrested': 1,
 'arrivals': 1,
 'arrived': 3,
 'arrives': 8,
 'arriving': 3,
 'article': 1,
 'asap': 1,
 'ashes': 1,
 'aside': 2,
 'ask': 1,
 'asked': 1,
 'asking': 2,
 'asks': 1,
 'asleep': 1,
 'ass': 7,
 'assassinated': 2,
 'assholes': 1,
 'assigned': 1,
 'assignment': 1,
 'assistant': 1,
 'associated': 1,
 'attempt': 1,
 'attention': 3,
 'autobahn': 1,
 'automatic': 3,
 'available': 1,
 'aware': 1,
 'away': 48,
 'awful': 3,
 'awhile': 2,
 'ba': 2,
 'backhanding': 1,
 'backing': 5,
 'backpack': 4,
 'backpacks': 2,
 'backs': 2,
 'backseat': 4,
 'backyard': 1,
 'bad': 5,
 'bag': 18,
 'bags': 2,
 'bail': 1,
 'bailing': 2,
 'bakery': 1,
 'balance': 2,
 'ball': 1,
 'balling': 1,
 'band': 1,
 'bang': 1,
 'banging': 1,
 'bank': 3,
 'banking': 1,
 'banks': 1,
 'bar': 1,
 'barely': 8,
 'bargain': 1,
 'barge': 9,
 'barn': 1,
 'base': 2,
 'basement': 2,
 'basic': 1,
 'basically': 1,
 'bastards': 1,
 'bathroom': 7,
 'battery': 1,
 'bb': 2,
 'bc': 1,
 'beach': 10,
 'bearing': 1,
 'bearings': 1,
 'beat': 22,
 'beats': 2,
 'beattocrap': 1,
 'bed': 9,
 'bedroom': 2,
 'beep': 2,
 'beeping': 2,
 'beggars': 1,
 'begging': 4,
 'begin': 2,
 'begins': 9,
 'behavior': 1,
 'behaviors': 1,
 'believe': 4,
 'believed': 6,
 'belongings': 1,
 'belt': 1,
 'beltway': 4,
 'bench': 1,
 'bends': 1,
 'beneath': 1,
 'berlin': 37,
 'better': 7,
 'bg': 1,
 'big': 17,
 'bigger': 3,
 'binders': 1,
 'binocular': 1,
 'birch': 1,
 'bit': 6,
 'bits': 1,
 'black': 15,
 'blade': 1,
 'blades': 1,
 'blam': 2,
 'blaming': 1,
 'blamm': 1,
 'blank': 2,
 'bleeding': 1,
 'blending': 1,
 'blinded': 1,
 'blindsided': 1,
 'blinking': 2,
 'blinks': 2,
 'block': 3,
 'blocked': 2,
 'blocking': 2,
 'blocks': 1,
 'blonde': 1,
 'blood': 6,
 'bloody': 2,
 'blow': 1,
 'blowing': 4,
 'blown': 3,
 'blows': 4,
 'blue': 3,
 'blueprint': 2,
 'blur': 3,
 'blurry': 1,
 'bmw': 4,
 'board': 2,
 'boarded': 1,
 'body': 9,
 'bogus': 1,
 'bomb': 1,
 'bombing': 1,
 'bone': 1,
 'book': 2,
 'boom': 1,
 'booooomm': 1,
 'booth': 3,
 'border': 2,
 'boss': 2,
 'bothering': 1,
 'bottle': 3,
 'boulevard': 4,
 'bouncing': 1,
 'bound': 1,
 'bourne': 455,
 'bourneabbott': 1,
 'bournephone': 5,
 'bournes': 29,
 'box': 5,
 'boxed': 1,
 'boxes': 1,
 'braced': 1,
 'braces': 1,
 'bracing': 2,
 'brakes': 3,
 'brandenburg': 1,
 'brass': 1,
 'breakdown': 1,
 'breaker': 1,
 'breaking': 1,
 'breaks': 1,
 'breath': 2,
 'breathe': 1,
 'breathing': 1,
 'breathless': 1,
 'brecker': 9,
 'bridge': 13,
 'briefcase': 8,
 'briefed': 1,
 'briefing': 1,
 'bright': 2,
 'bring': 2,
 'briskly': 2,
 'bristles': 1,
 'broadcoasting': 1,
 'bruise': 1,
 'budget': 1,
 'building': 20,
 'bulge': 1,
 'bulging': 1,
 'bulkhead': 2,
 'bullet': 1,
 'bulletin': 1,
 'bullets': 2,
 'bullpen': 5,
 'bumpers': 1,
 'bunch': 1,
 'bundle': 1,
 'burly': 1,
 'burn': 1,
 'burning': 2,
 'burst': 1,
 'bus': 3,
 'business': 3,
 'bust': 1,
 'busted': 1,
 'busy': 7,
 'button': 1,
 'buttoning': 1,
 'buy': 7,
 'buying': 2,
 'buzzkill': 1,
 'cab': 24,
 'cabbie': 3,
 'cabby': 1,
 'cabin': 1,
 'cabinet': 3,
 'cable': 1,
 'cabled': 1,
 'caffeine': 1,
 'calculated': 1,
 'calendar': 1,
 'caliber': 1,
 'called': 3,
 'calling': 1,
 'calls': 1,
 'calm': 4,
 'calmly': 2,
 'calms': 1,
 'came': 10,
 'camera': 3,
 'cameras': 1,
 'campground': 1,
 'canhow': 1,
 'canvas': 1,
 'cap': 1,
 'captain': 2,
 'caption': 1,
 'car': 51,
 'carabinieri': 5,
 'carabinieris': 1,
 'card': 1,
 'cards': 2,
 'care': 3,
 'careening': 1,
 'careful': 2,
 'carefully': 2,
 'careless': 1,
 'carnage': 1,
 'carries': 1,
 'carrying': 6,
 'cars': 20,
 'carves': 1,
 'cascading': 1,
 'case': 7,
 'cash': 6,
 'caspian': 1,
 'caspiexpetroleum': 1,
 'cast': 1,
 'casual': 2,
 'casually': 1,
 'catches': 1,
 'catching': 3,
 'caught': 2,
 'cause': 1,
 'caution': 1,
 'caves': 1,
 'cd': 1,
 'cell': 7,
 'cellphone': 2,
 'cement': 1,
 'center': 3,
 'ceo': 1,
 'certainly': 1,
 'chair': 9,
 'chairs': 1,
 'chalkboard': 1,
 'championed': 1,
 'championship': 1,
 'chance': 1,
 'change': 4,
 'changed': 3,
 'changes': 2,
 'chaos': 2,
 'charge': 7,
 'charges': 2,
 'charmed': 1,
 'chase': 2,
 'chasing': 1,
 'chatter': 1,
 'chatting': 1,
 'check': 6,
 'checked': 2,
 'checking': 13,
 'checkoff': 1,
 'checks': 12,
 'cherbourg': 1,
 'childlike': 1,
 'chill': 1,
 'chilly': 1,
 'chinese': 2,
 'chirping': 1,
 'choice': 5,
 'choices': 1,
 'chokehold': 1,
 'chokes': 1,
 'chop': 2,
 'choreographed': 1,
 'christ': 2,
 'chucked': 1,
 'chugging': 2,
 'ci': 2,
 'cia': 7,
 'cigarette': 1,
 'cigarettes': 1,
 'cinch': 1,
 'circles': 2,
 'city': 4,
 'civvies': 1,
 'claimed': 1,
 'clean': 7,
 'cleaning': 1,
 'clear': 9,
 'clearance': 1,
 'clearing': 1,
 'clearly': 5,
 'clears': 2,
 'clenching': 1,
 'clerk': 8,
 'clerks': 1,
 'clever': 1,
 'click': 6,
 'clicking': 1,
 'clicks': 1,
 'cliff': 1,
 'climb': 1,
 'climbing': 4,
 'clip': 3,
 'clipping': 1,
 'clock': 4,
 'clogging': 1,
 'cloned': 2,
 'close': 13,
 'closed': 3,
 'closequarter': 1,
 'closer': 4,
 'closes': 2,
 'closing': 3,
 'clothes': 7,
 'club': 3,
 'clubhouse': 1,
 'clue': 1,
 'cluster': 1,
 'clutter': 1,
 'cluttered': 1,
 'coal': 2,
 'coat': 7,
 'code': 2,
 'coding': 1,
 'coffee': 5,
 'cold': 4,
 'colonial': 1,
 'colored': 1,
 'come': 18,
 'comes': 10,
 'comfortable': 1,
 'coming': 30,
 'comm': 2,
 'command': 1,
 'commanders': 1,
 'common': 1,
 'commotion': 1,
 'communications': 2,
 'commuters': 1,
 'companies': 1,
 'company': 3,
 'comparison': 1,
 'complaining': 1,
 'compressing': 1,
 'compulsive': 1,
 'computer': 7,
 'computers': 2,
 'comrade': 1,
 'concentrating': 1,
 'concentric': 1,
 'concerned': 2,
 'concerning': 1,
 'concierge': 3,
 'concrete': 3,
 'condition': 1,
 'conditions': 1,
 'conductors': 1,
 'conferring': 2,
 'confidentially': 1,
 'confirm': 1,
 'confusion': 3,
 'conjunction': 1,
 'conjure': 1,
 'conklin': 29,
 'conklins': 4,
 'connections': 1,
 'consider': 1,
 'considering': 1,
 'consist': 1,
 'console': 1,
 'construction': 2,
 'consulate': 2,
 'contact': 3,
 'contd': 63,
 'continents': 1,
 'continues': 3,
 'continuing': 8,
 'contract': 1,
 'control': 2,
 'controlled': 1,
 'converge': 1,
 'conversation': 3,
 'convinced': 1,
 'convoy': 1,
 'coo': 1,
 'cool': 3,
 'cooly': 1,
 'coordinate': 1,
 'cop': 5,
 'cops': 22,
 'corner': 13,
 'corral': 1,
 'corridor': 1,
 'cots': 1,
 'cottage': 1,
 'coughing': 1,
 'counter': 2,
 'countering': 1,
 'counterintelligence': 2,
 'counting': 1,
 'country': 1,
 'countryside': 1,
 'couple': 4,
 'course': 1,
 'courtesy': 1,
 'cover': 9,
 'covered': 2,
 'covering': 1,
 'coworkers': 1,
 'cranking': 1,
 'cranks': 1,
 'crap': 2,
 'crash': 2,
 'crashes': 1,
 'crashing': 2,
 'crazy': 4,
 'credentials': 1,
 'credit': 1,
 'creepy': 1,
 'crestfallen': 1,
 'crewcut': 1,
 'crime': 2,
 'crinkles': 1,
 'crisp': 1,
 'crissake': 1,
 'cronin': 81,
 'croninradio': 1,
 'cross': 2,
 'crosses': 3,
 'crossing': 2,
 'crouches': 1,
 'crowd': 7,
 'crowded': 2,
 'cruising': 1,
 'crush': 1,
 'crushed': 1,
 'crushes': 1,
 'crying': 4,
 'cuffed': 2,
 'cuffs': 2,
 'cup': 2,
 'curb': 3,
 'curiosity': 1,
 'curious': 1,
 'curtains': 2,
 'curving': 1,
 'customs': 1,
 'cut': 2,
 'cuts': 2,
 'cutting': 3,
 'cyrillic': 2,
 'da': 1,
 'dad': 1,
 'damn': 2,
 'dangerous': 3,
 'dangle': 1,
 'daniel': 1,
 'danka': 1,
 'danny': 8,
 'dark': 16,
 'darkened': 1,
 'darkness': 2,
 'dash': 2,
 'dashboards': 1,
 'data': 3,
 'database': 1,
 'date': 2,
 'david': 1,
 'day': 4,
 'days': 2,
 'dead': 22,
 'deadly': 1,
 'deal': 3,
 'dealing': 1,
 'death': 4,
 'debrief': 2,
 'debriefed': 1,
 'decent': 1,
 'decide': 2,
 'decided': 1,
 'decision': 1,
 'decives': 1,
 'deck': 2,
 'deep': 2,
 'defensive': 2,
 'defiant': 1,
 'definitely': 1,
 'definitive': 3,
 'delta': 12,
 'deltas': 1,
 'departure': 2,
 'departures': 1,
 'depression': 1,
 'deputy': 2,
 'descend': 1,
 'deserted': 2,
 'desk': 17,
 'desolate': 2,
 'desperation': 2,
 'destiny': 1,
 'destroy': 1,
 'destroys': 1,
 'detailed': 1,
 'details': 2,
 'detained': 1,
 'determined': 1,
 'detonation': 1,
 'device': 2,
 'diagnosis': 1,
 'diagnostic': 1,
 'dial': 1,
 'dialed': 1,
 'did': 15,
 'didi': 1,
 'didjason': 1,
 'didnt': 11,
 'die': 1,
 'died': 3,
 'diei': 1,
 'dies': 1,
 'different': 2,
 'digital': 1,
 'digs': 1,
 'direct': 1,
 'directions': 1,
 'directly': 1,
 'director': 2,
 'disappear': 6,
 'disappearing': 1,
 'disappears': 3,
 'disappointed': 1,
 'disaster': 3,
 'discreet': 2,
 'discreetly': 2,
 'disembark': 1,
 'disembarking': 2,
 'disembarks': 1,
 'disgusted': 1,
 'dishes': 1,
 'disputing': 1,
 'distance': 4,
 'distinctive': 1,
 'distribution': 1,
 'disturbance': 1,
 'ditch': 1,
 'dividing': 1,
 'diving': 1,
 'dobermans': 2,
 'doctor': 1,
 'document': 1,
 'dod': 3,
 'dodging': 1,
 'dods': 1,
 'does': 9,
 'doesnt': 11,
 'doing': 20,
 'dollar': 2,
 'dollars': 4,
 'domestic': 1,
 'dominant': 1,
 'dominates': 1,
 'donnie': 1,
 'dont': 42,
 'door': 31,
 'doorbell': 1,
 'doorman': 2,
 'doors': 9,
 'doorway': 3,
 'double': 1,
 'doublecrossed': 1,
 'doubt': 1,
 'downhill': 1,
 'downs': 1,
 'dozens': 1,
 'drab': 1,
 'drawn': 4,
 'dread': 2,
 'dream': 4,
 'dresser': 1,
 'drifting': 2,
 'drill': 3,
 'drills': 2,
 'drink': 1,
 'drinks': 1,
 'drive': 3,
 'driver': 20,
 'driverphone': 1,
 'drivers': 3,
 'drives': 4,
 'driving': 12,
 'drone': 1,
 'drop': 4,
 'dropping': 3,
 'drops': 3,
 'drugs': 1,
 'drunk': 4,
 'ducked': 1,
 'ducking': 1,
 'duct': 2,
 'dude': 2,
 'dudes': 6,
 'duffel': 1,
 'duffle': 3,
 'dumping': 1,
 'dunk': 1,
 'ear': 1,
 'earlier': 1,
 'early': 5,
 'earpiece': 3,
 'ears': 1,
 'easing': 1,
 'east': 2,
 'easy': 2,
 'eat': 1,
 'edge': 2,
 'edges': 1,
 'ef': 1,
 'effective': 1,
 'effort': 1,
 'effortless': 1,
 'electric': 1,
 'electrical': 3,
 'element': 1,
 'elevator': 8,
 'eluded': 1,
 'embankment': 1,
 'embarrassed': 2,
 'emerge': 1,
 'emergency': 1,
 'emerges': 1,
 'emerging': 2,
 'emphatic': 1,
 'emptied': 1,
 'end': 10,
 'ends': 2,
 'energy': 1,
 'engine': 1,
 'engines': 1,
 'english': 2,
 'enjoy': 1,
 'enter': 2,
 'entering': 1,
 'enters': 10,
 'entombed': 1,
 'entrance': 2,
 'entrances': 1,
 'equal': 1,
 'equally': 1,
 'escape': 1,
 'escort': 3,
 'ether': 1,
 'europe': 2,
 'european': 1,
 'euros': 1,
 'evac': 2,
 'event': 1,
 'everybody': 3,
 'everyday': 1,
 'evidence': 1,
 'ex': 1,
 'exactly': 7,
 'exceptionally': 1,
 'excited': 2,
 'excuse': 1,
 'exfil': 6,
 'exit': 2,
 'exiting': 1,
 'exits': 3,
 'exnavyseal': 1,
 'expensive': 1,
 'expert': 1,
 'explosion': 1,
 'explosive': 3,
 'express': 1,
 'extreme': 1,
 'extremely': 2,
 'eye': 6,
 'eyes': 25,
 'facade': 2,
 'facades': 1,
 'face': 22,
 'faces': 6,
 'fact': 2,
 'factories': 1,
 'factoring': 1,
 'fade': 2,
 'fading': 1,
 'fail': 1,
 'failed': 1,
 'failing': 1,
 'fair': 1,
 'fake': 1,
 'fakes': 1,
 'faking': 1,
 'fall': 3,
 'fallen': 1,
 'falling': 2,
 'falls': 1,
 'familiar': 2,
 'families': 1,
 'family': 5,
 'fan': 1,
 'fanning': 1,
 'far': 7,
 'fast': 13,
 'faster': 1,
 'father': 3,
 'fatherly': 1,
 'fault': 1,
 'faux': 1,
 'favorite': 1,
 'fax': 2,
 'fear': 4,
 'feed': 2,
 'feeding': 1,
 'feel': 4,
 'feels': 1,
 'feet': 3,
 'fell': 3,
 'felt': 4,
 'fender': 1,
 'fenders': 1,
 'fernsehturm': 1,
 'ferry': 2,
 'field': 6,
 'fiftyeuro': 1,
 'fighting': 1,
 'fights': 1,
 'figure': 3,
 'file': 8,
 'filei': 1,
 'files': 11,
 'fileyour': 1,
 'filing': 1,
 'filling': 1,
 'fills': 1,
 'final': 5,
 'finally': 11,
 'financial': 2,
 'finding': 3,
 'finds': 1,
 'fine': 2,
 'finger': 4,
 'fingerprint': 3,
 'fingers': 1,
 'finish': 2,
 'finished': 1,
 'finishes': 1,
 'finishing': 1,
 'fired': 1,
 'fires': 1,
 'firing': 4,
 'firmly': 1,
 'fishing': 2,
 'fishtailing': 1,
 'fix': 2,
 'fixed': 3,
 'flag': 2,
 'flames': 1,
 'flanking': 1,
 'flash': 2,
 'flashback': 7,
 'flashes': 3,
 'flashing': 2,
 'flat': 2,
 'flatout': 1,
 'fleeing': 1,
 'flexcuffs': 2,
 'flicker': 3,
 'flickering': 1,
 'flicking': 1,
 'flies': 1,
 'flight': 1,
 'flights': 1,
 'flimsy': 1,
 'flips': 2,
 'floated': 1,
 'flood': 1,
 'flooding': 1,
 'floor': 10,
 'flooring': 1,
 'floors': 1,
 'flow': 1,
 'fly': 1,
 'flying': 2,
 'focused': 3,
 'focusing': 1,
 'foggy': 1,
 'folder': 1,
 'folding': 2,
 'folks': 1,
 'follow': 3,
 'followed': 1,
 'following': 2,
 'follows': 3,
 'food': 4,
 'fooled': 2,
 'foot': 4,
 'footing': 1,
 'footlocker': 2,
 'footpath': 2,
 'footstep': 1,
 'footsteps': 1,
 'force': 2,
 'foreboding': 1,
 'forehead': 1,
 'forest': 1,
 'forever': 1,
 'forget': 3,
 'forging': 1,
 'forgive': 1,
 'forgotten': 2,
 'fork': 2,
 'forlorn': 1,
 'forties': 1,
 'fortune': 1,
 'forward': 6,
 'fourgam': 1,
 'fourth': 1,
 'frantic': 1,
 'freaked': 3,
 'free': 4,
 'freezes': 1,
 'freezing': 2,
 'frequency': 1,
 'fresh': 4,
 'fridge': 1,
 'friendly': 1,
 'friends': 2,
 'frowns': 1,
 'frozen': 2,
 'frustration': 1,
 'fry': 1,
 'fuck': 7,
 'fucking': 1,
 'fully': 3,
 'fumbling': 1,
 'funky': 1,
 'furious': 1,
 'fury': 2,
 'fusty': 1,
 'futile': 1,
 'gables': 1,
 'gadgetry': 1,
 'gaining': 2,
 'game': 4,
 'gangster': 1,
 'gap': 1,
 'gas': 5,
 'gasolinestoked': 1,
 'gasping': 1,
 'gassoaked': 1,
 'gate': 1,
 'gathered': 2,
 'gathering': 1,
 'gauge': 1,
 'gaze': 2,
 'gear': 9,
 'gears': 1,
 'geneva': 1,
 'genevai': 1,
 'genoa': 1,
 'gentlemen': 1,
 'german': 7,
 'germans': 1,
 'germany': 2,
 'gestures': 3,
 'gets': 7,
 'getting': 16,
 'gift': 1,
 'gimme': 1,
 'girl': 6,
 'girlfriend': 1,
 'girls': 1,
 'given': 1,
 'gives': 4,
 'giving': 3,
 'glacial': 1,
 'glance': 1,
 'glanced': 1,
 'glances': 1,
 'glancing': 8,
 ...}
</pre>

```python
import operator

sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)[:5]
```

<pre>
[('bourne', 455),
 ('pamela', 199),
 ('abbott', 128),
 ('hes', 100),
 ('kirill', 93)]
</pre>

```python
plt.hist(list(word_count_dict.values()), bins=150)
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX0AAAD5CAYAAADLL+UrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAPLklEQVR4nO3cb8jdZ33H8fdnac3Kalm7pl2WhN2ZZLC0bNHeZIWO0c2xxj8s9YGQwmwelEVKZcqEkShM9yCgY+omrIW4lqZMLQGVBtvOZZlDBp3xTq02acwabWezhOZ2MoxPujX97sG5Yo53T3P/S07qfb1fcDi/8z3X7/yu3/fB5z73dX7npKqQJPXh5y71BCRJ42PoS1JHDH1J6oihL0kdMfQlqSOGviR15LLZBiRZAzwE/DLwCrCrqv42yUeBPwGm29APVdVjbZ8dwF3AGeBPq+orrX4T8CBwBfAY8P6a5ZrRa6+9tiYmJuZ9YpLUs4MHD/6gqlbMrM8a+sDLwAer6skkbwQOJtnXnvtUVf318OAk64EtwA3ArwD/nOTXq+oMcB+wDfh3BqG/CXj8fAefmJhgampqDtOUJJ2V5D9H1Wdd3qmqk1X1ZNs+DRwBVp1nl83Aw1X1UlU9BxwDNiZZCVxVVU+0d/cPAbfP7zQkSYsxrzX9JBPAm4Gvt9L7knw7yQNJrm61VcALQ7sdb7VVbXtmXZI0JnMO/SRXAl8APlBVP2KwVPMmYANwEvjE2aEjdq/z1Ecda1uSqSRT09PTo4ZIkhZgTqGf5HIGgf/ZqvoiQFW9WFVnquoV4DPAxjb8OLBmaPfVwIlWXz2i/ipVtauqJqtqcsWKV30OIUlaoFlDP0mA+4EjVfXJofrKoWHvAg617b3AliTLk6wF1gEHquokcDrJze017wQeuUDnIUmag7lcvXML8B7g6SRPtdqHgDuSbGCwRPM88F6AqjqcZA/wDIMrf+5pV+4A3M25SzYfZ5YrdyRJF1Ze7z+tPDk5WV6yKUnzk+RgVU3OrPuNXEnqiKEvSR1Z0qE/sf1RJrY/eqmnIUmvG0s69CVJP83Ql6SOGPqS1BFDX5I6YuhLUkcMfUnqiKEvSR0x9CWpI4a+JHXE0Jekjhj6ktQRQ1+SOmLoS1JHDH1J6oihL0kdMfQlqSOGviR1xNCXpI4Y+pLUEUNfkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6YuhLUkcMfUnqiKEvSR2ZNfSTrEny1SRHkhxO8v5WvybJviTPtvurh/bZkeRYkqNJbhuq35Tk6fbcp5Pk4pyWJGmUubzTfxn4YFX9BnAzcE+S9cB2YH9VrQP2t8e057YANwCbgHuTLGuvdR+wDVjXbpsu4LlIkmYxa+hX1cmqerJtnwaOAKuAzcDuNmw3cHvb3gw8XFUvVdVzwDFgY5KVwFVV9URVFfDQ0D6SpDGY15p+kgngzcDXgeur6iQM/jAA17Vhq4AXhnY73mqr2vbM+qjjbEsylWRqenp6PlOUJJ3HnEM/yZXAF4APVNWPzjd0RK3OU391sWpXVU1W1eSKFSvmOkVJ0izmFPpJLmcQ+J+tqi+28ottyYZ2f6rVjwNrhnZfDZxo9dUj6pKkMZnL1TsB7geOVNUnh57aC2xt21uBR4bqW5IsT7KWwQe2B9oS0OkkN7fXvHNoH0nSGFw2hzG3AO8Bnk7yVKt9CPgYsCfJXcD3gXcDVNXhJHuAZxhc+XNPVZ1p+90NPAhcATzebpKkMZk19Kvq3xi9Hg/w1tfYZyewc0R9CrhxPhOUJF04fiNXkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6YuhLUkcMfUnqiKEvSR0x9CWpI4a+JHXE0Jekjhj6ktQRQ1+SOmLoS1JHDH1J6oihL0kdMfQlqSOGviR1xNCXpI4Y+pLUEUNfkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6YuhLUkdmDf0kDyQ5leTQUO2jSf4ryVPt9vah53YkOZbkaJLbhuo3JXm6PffpJLnwpyNJOp+5vNN/ENg0ov6pqtrQbo8BJFkPbAFuaPvcm2RZG38fsA1Y126jXlOSdBHNGvpV9TXgh3N8vc3Aw1X1UlU9BxwDNiZZCVxVVU9UVQEPAbcvcM6SpAVazJr++5J8uy3/XN1qq4AXhsYcb7VVbXtmfaQk25JMJZmanp5exBQlScMWGvr3AW8CNgAngU+0+qh1+jpPfaSq2lVVk1U1uWLFigVOUZI004JCv6perKozVfUK8BlgY3vqOLBmaOhq4ESrrx5RlySN0YJCv63Rn/Uu4OyVPXuBLUmWJ1nL4APbA1V1Ejid5OZ21c6dwCOLmLckaQEum21Aks8DtwLXJjkOfAS4NckGBks0zwPvBaiqw0n2AM8ALwP3VNWZ9lJ3M7gS6Arg8XaTJI3RrKFfVXeMKN9/nvE7gZ0j6lPAjfOanSTpgvIbuZLUEUNfkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6YuhLUkcMfUnqiKEvSR0x9CWpI4a+JHXE0Jekjhj6ktQRQ1+SOmLoS1JHDH1J6oihL0kdMfQlqSOGviR1xNCXpI4Y+pLUEUNfkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6MmvoJ3kgyakkh4Zq1yTZl+TZdn/10HM7khxLcjTJbUP1m5I83Z77dJJc+NORJJ3PXN7pPwhsmlHbDuyvqnXA/vaYJOuBLcANbZ97kyxr+9wHbAPWtdvM15QkXWSzhn5VfQ344YzyZmB3294N3D5Uf7iqXqqq54BjwMYkK4GrquqJqirgoaF9JEljstA1/eur6iRAu7+u1VcBLwyNO95qq9r2zPpISbYlmUoyNT09vcApSpJmutAf5I5ap6/z1Eeqql1VNVlVkytWrLhgk5Ok3i009F9sSza0+1OtfhxYMzRuNXCi1VePqEuSxmihob8X2Nq2twKPDNW3JFmeZC2DD2wPtCWg00lublft3Dm0jyRpTC6bbUCSzwO3AtcmOQ58BPgYsCfJXcD3gXcDVNXhJHuAZ4CXgXuq6kx7qbsZXAl0BfB4u0mSxmjW0K+qO17jqbe+xvidwM4R9SngxnnNTpJ0QfmNXEnqiKEvSR0x9CWpI4a+JHXE0Jekjhj6ktQRQ1+SOmLoS1JHDH1J6oihL0kdMfQlqSOGviR1xNCXpI4Y+pLUEUNfkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6YuhLUkcMfUnqiKEvSR0x9CWpI4a+JHXE0Jekjhj6ktQRQ1+SOmLoS1JHDH1J6oihL0kdWVToJ3k+ydNJnkoy1WrXJNmX5Nl2f/XQ+B1JjiU5muS2xU5ekjQ/F+Kd/u9V1YaqmmyPtwP7q2odsL89Jsl6YAtwA7AJuDfJsgtwfEnSHF2M5Z3NwO62vRu4faj+cFW9VFXPAceAjRfh+JKk17DY0C/gn5IcTLKt1a6vqpMA7f66Vl8FvDC07/FWe5Uk25JMJZmanp5e5BQlSWddtsj9b6mqE0muA/Yl+c55xmZErUYNrKpdwC6AycnJkWMkSfO3qHf6VXWi3Z8CvsRguebFJCsB2v2pNvw4sGZo99XAicUcX5I0PwsO/SS/kOSNZ7eBPwQOAXuBrW3YVuCRtr0X2JJkeZK1wDrgwEKPL0mav8Us71wPfCnJ2df5XFX9Y5JvAHuS3AV8H3g3QFUdTrIHeAZ4Gbinqs4savaSpHlZcOhX1feA3xpR/2/gra+xz05g50KPKUlaHL+RK0kdMfQlqSOGviR1xNCXpI4Y+pLUEUNfkjpi6EtSRwx9SeqIoS9JHTH0Jakjhr4kdcTQl6SOGPqS1BFDX5I6YuhLUkcMfUnqiKEvSR0x9CWpI4a+JHXE0Jekjhj6ktQRQ1+SOtJF6E9sf5SJ7Y9e6mlI0iXXRehLkgYMfUnqiKEvSR0x9CWpI4a+JHXE0JekjnQV+l66Kal3XYW+JPXO0Jekjhj6ktSRsYd+kk1JjiY5lmT7uI8Pru1L6tdYQz/JMuDvgLcB64E7kqwf5xyGGf6SenPZmI+3EThWVd8DSPIwsBl4Zszz+Ckzg//5j73jEs1Eki6ucYf+KuCFocfHgd8e8xxmNdu7/7N/FPxjIelnzbhDPyNq9apByTZgW3v44yRHF3Csa4EfLGC/WeXj86u/Dly0XvwMshfn2ItzlmIvfnVUcdyhfxxYM/R4NXBi5qCq2gXsWsyBkkxV1eRiXmOpsBfn2Itz7MU5PfVi3FfvfANYl2RtkjcAW4C9Y56DJHVrrO/0q+rlJO8DvgIsAx6oqsPjnIMk9WzcyztU1WPAY2M41KKWh5YYe3GOvTjHXpzTTS9S9arPUSVJS5Q/wyBJHVmSof96+KmHcUryQJJTSQ4N1a5Jsi/Js+3+6qHndrTeHE1y26WZ9cWRZE2SryY5kuRwkve3enf9SPLzSQ4k+VbrxV+2ene9gMEvAiT5ZpIvt8dd9oGqWlI3Bh8Qfxf4NeANwLeA9Zd6Xhf5nH8XeAtwaKj2V8D2tr0d+HjbXt96shxY23q17FKfwwXsxUrgLW37jcB/tHPurh8MvhdzZdu+HPg6cHOPvWjn92fA54Avt8dd9mEpvtP/yU89VNX/Amd/6mHJqqqvAT+cUd4M7G7bu4Hbh+oPV9VLVfUccIxBz5aEqjpZVU+27dPAEQbfBO+uHzXw4/bw8nYrOuxFktXAO4C/Hyp31wdYmss7o37qYdUlmsuldH1VnYRBEALXtXo3/UkyAbyZwTvcLvvRljSeAk4B+6qq1178DfDnwCtDtR77sCRDf04/9dCxLvqT5ErgC8AHqupH5xs6orZk+lFVZ6pqA4Nvv29McuN5hi/JXiR5J3Cqqg7OdZcRtZ/5Ppy1FEN/Tj/10IEXk6wEaPenWn3J9yfJ5QwC/7NV9cVW7rYfAFX1P8C/Apvorxe3AH+U5HkGy72/n+Qf6K8PwNIMfX/qYWAvsLVtbwUeGapvSbI8yVpgHXDgEszvokgS4H7gSFV9cuip7vqRZEWSX2zbVwB/AHyHznpRVTuqanVVTTDIg3+pqj+msz78xKX+JPli3IC3M7hq47vAhy/1fMZwvp8HTgL/x+Bdyl3ALwH7gWfb/TVD4z/cenMUeNulnv8F7sXvMPhX/NvAU+329h77Afwm8M3Wi0PAX7R6d70YOr9bOXf1Tpd98Bu5ktSRpbi8I0l6DYa+JHXE0Jekjhj6ktQRQ1+SOmLoS1JHDH1J6oihL0kd+X9VUd6jdCnj3gAAAABJRU5ErkJggg=="/>

-----


## 4) text mining


### 4-1) Frequency analysis by word


##### word cloud visualization



```python
from collections import Counter

import random
import pytagcloud
import webbrowser

ranked_tags = Counter(word_count_dict).most_common(25)
taglist = pytagcloud.make_tags(sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)[:40], maxsize=60)
pytagcloud.create_tag_image(taglist, 'wordcloud_example.jpg', 
                            rectangular=False)

from IPython.display import Image
Image(filename='wordcloud_example.jpg')
```

<pre>
pygame 2.1.2 (SDL 2.0.18, Python 3.8.8)
Hello from the pygame community. https://www.pygame.org/contribute.html
</pre>
<pre>
<IPython.core.display.Image object>
</pre>
-----


### 4-2) Visualize key words by scene


##### TF-IDF Conversion



```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)
```


```python
tf_idf_vect.shape
```

<pre>
(320, 2850)
</pre>

```python
print(tf_idf_vect[0])
```

<pre>
  (0, 2788)	0.19578974958217082
  (0, 2763)	0.27550455848587985
  (0, 2412)	0.1838379942679887
  (0, 2387)	0.3109660261831164
  (0, 1984)	0.2902223973596984
  (0, 1978)	0.3109660261831164
  (0, 1898)	0.27550455848587985
  (0, 1673)	0.2902223973596984
  (0, 1366)	0.21520447034992146
  (0, 1251)	0.19855583314180728
  (0, 1001)	0.2340173008390438
  (0, 974)	0.2902223973596984
  (0, 874)	0.27550455848587985
  (0, 798)	0.1906694714764746
  (0, 237)	0.08646242181596513
  (0, 125)	0.26408851574819875
</pre>

```python
print(tf_idf_vect[0].toarray().shape)
print(tf_idf_vect[0].toarray())
```

<pre>
(1, 2850)
[[0. 0. 0. ... 0. 0. 0.]]
</pre>
##### Vector: word mapping



```python
invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100]+'..')
```

<pre>
{1898: 'raining', 1366: 'light', 2387: 'strobes', 2763: 'wet', 1001: 'glass', 1978: 'rhythmic', 1673..
</pre>
##### Key Word Extraction - Top 3 TF-IDF



```python
np.argsort(tf_idf_vect[0].toarray())[0][-3:]
```

<pre>
array([1984, 2387, 1978], dtype=int64)
</pre>

```python
np.argsort(tf_idf_vect.toarray())[:, -3:]
```

<pre>
array([[1984, 2387, 1978],
       [1297, 1971, 1097],
       [1693, 2221,  968],
       [ 690,  299, 1482],
       [2823, 1951, 1454],
       [2218, 2815, 1454],
       [2038,  737, 2418],
       [ 852, 2761, 2570],
       [2022,  156, 1352],
       [2250, 2241, 1454],
       [ 342,  321, 2188],
       [ 614, 1557, 1534],
       [ 535, 1884, 1614],
       [2188,  139,   20],
       [ 503,  730, 1458],
       [2790, 2384,  724],
       [ 169,  915, 2444],
       [1905, 1259,   53],
       [2566, 1335,  828],
       [2300,  281, 1702],
       [2503, 1502, 2567],
       [ 794, 1454, 1018],
       [ 698, 2559, 1252],
       [1871,  237, 1454],
       [ 204,  911, 2591],
       [ 237,  596, 1454],
       [  52,  941, 1036],
       [ 211, 1156,  206],
       [1193, 2712, 1454],
       [  52, 1809, 2462],
       [ 237, 1454,  702],
       [2130,  237, 1454],
       [1995, 1890,  321],
       [1011,  259, 1454],
       [1985, 2216, 1819],
       [ 420, 2276, 1454],
       [2019, 1103, 2059],
       [2177,  353, 1252],
       [1642, 1291, 2797],
       [1454, 1012, 2439],
       [2000, 2429, 2051],
       [2111, 1559, 2661],
       [1737, 1291, 2429],
       [1116, 2079,   46],
       [2634,  697,  392],
       [1723, 1015, 1071],
       [1443, 2700, 2486],
       [ 332, 2354,  786],
       [1165,  543,  975],
       [1169, 2434, 1986],
       [1509, 2486, 2335],
       [1001, 2686, 1509],
       [1289,  854,  219],
       [2758, 1687, 1460],
       [ 544, 1031,  854],
       [2686,  741,  726],
       [ 790,  625, 2794],
       [ 955,  367, 2583],
       [ 337, 1238, 2686],
       [ 340, 1634, 2490],
       [1238, 2248, 2686],
       [ 624,  789, 2538],
       [ 579,  341,  805],
       [ 543,  270,  577],
       [  68,  146,  708],
       [1291, 1100, 1069],
       [ 162, 1291, 1033],
       [ 300, 2786, 1900],
       [ 543, 1687, 2380],
       [ 431,  782,  917],
       [2006,  128, 1751],
       [1687,  543, 2485],
       [ 552, 1866,  235],
       [2695,  762, 1102],
       [ 265, 2641,   56],
       [ 612, 1691,  441],
       [1654,  692, 1363],
       [2136,  243, 1688],
       [   7, 2848,    6],
       [2596, 1687,    4],
       [2635, 1483,  841],
       [ 653, 2650, 1882],
       [1687,  481,    4],
       [1715, 1198, 1633],
       [2219, 2482,  169],
       [1541, 2423,  538],
       [ 322, 1633, 1198],
       [1460,    4,  414],
       [1307,  270, 1593],
       [1407, 1687,  543],
       [ 237,  491, 1593],
       [ 792,  348, 2665],
       [1055, 1747, 1593],
       [2010, 1868, 1403],
       [  80, 2415, 2611],
       [2361, 1747, 1593],
       [1687, 1594, 1688],
       [1310, 1593,  143],
       [2825, 1688, 1594],
       [1460,    4, 1687],
       [   4,  508, 2848],
       [1745,  983, 1612],
       [ 180,  491, 1690],
       [1496,  132, 1600],
       [1687, 1105, 1600],
       [2849, 2211, 2378],
       [ 173, 1687, 1600],
       [1757, 2379,  652],
       [2368, 2086, 1609],
       [   4, 1312, 1572],
       [1725, 1863, 1418],
       [1873, 1274,  270],
       [1291, 2718,  683],
       [ 108, 1033, 2421],
       [1104,  650,  382],
       [ 491,  237, 1248],
       [2768, 1561, 1247],
       [ 939,  237, 1248],
       [1561, 2008, 1721],
       [ 883,  623, 1248],
       [ 944, 1561, 1246],
       [ 855, 1248, 1055],
       [ 458, 1914, 1250],
       [ 930,  966, 2550],
       [2380, 2516, 2625],
       [ 670,  228,   81],
       [2660, 1780,  902],
       [2371, 2553, 2515],
       [1716,  706,   93],
       [ 926,  780, 2848],
       [1585,  659, 1118],
       [1967, 1180,  208],
       [1992,  268, 2741],
       [ 237, 1521, 2377],
       [2178, 2003, 1296],
       [ 608, 2504,  970],
       [2849,  237,  706],
       [1655, 1049, 2700],
       [1160, 1322, 2197],
       [2375, 2223, 1776],
       [2475,  725, 2381],
       [ 703,  258,  481],
       [2646, 2229,  467],
       [ 237, 1059, 2067],
       [1828, 1669, 2112],
       [ 741, 1390,  783],
       [1562, 2407,  543],
       [ 295, 1269, 1906],
       [2437,  771,  703],
       [1291, 2245,  260],
       [2368, 2008, 2437],
       [ 891, 1400, 1960],
       [  62, 2185, 1676],
       [1465,  919,  241],
       [2658, 1358, 1582],
       [1091,  595, 1002],
       [ 738, 1424, 1600],
       [2726, 1600, 1687],
       [2451, 2787,  273],
       [   4,  680, 1687],
       [ 755,  401,  155],
       [1593,    4, 1413],
       [ 607, 1600, 1817],
       [2329, 1144,  254],
       [  80, 1601, 2582],
       [ 606, 1600, 2582],
       [1434,  630, 1381],
       [2551, 1350,  173],
       [ 727, 2137, 2702],
       [1931, 1600, 1307],
       [1692,  372, 1893],
       [1830,  237, 1600],
       [ 504, 2344,  606],
       [2408,  819,  763],
       [2623, 1342, 1228],
       [1033, 1319,  890],
       [ 521,  481,  293],
       [  10,  826, 2586],
       [ 167,  237, 1600],
       [2014, 1687, 2137],
       [1700,  218, 2120],
       [1225, 2750,  942],
       [ 442, 2848,    4],
       [ 981,  258, 1918],
       [2323, 2355, 2848],
       [ 221,    4, 2848],
       [   1,    0, 1160],
       [   8, 2014,  411],
       [1984,  694, 1552],
       [1775, 2133,  829],
       [1441, 2053, 2205],
       [  55,  275,  267],
       [1647, 1306, 2014],
       [2415, 2416,  878],
       [2780,  650, 1171],
       [2329,  673,  256],
       [ 833, 1111, 1813],
       [ 366,  543, 2485],
       [1896, 2495, 1196],
       [2604,  651, 1389],
       [1344, 1272,  882],
       [1558, 1517, 1590],
       [1697, 1661, 2192],
       [1444,  319, 2507],
       [2615, 2318,  255],
       [2746, 2477, 2441],
       [1730, 2014, 1988],
       [ 124,  133, 2435],
       [1439, 1712, 1700],
       [1282, 1137,  896],
       [2788, 1548,  152],
       [1659, 1340, 2015],
       [1486, 2441, 2011],
       [1555, 2767,  349],
       [ 204,  762,  784],
       [1894,  407, 2014],
       [ 796,   88, 2077],
       [2728, 2323, 2659],
       [1291, 2011, 1110],
       [ 577,  374, 2485],
       [2300, 1830,   11],
       [1160, 2485,    4],
       [1147, 2777,  954],
       [2284,  973,  823],
       [ 116, 1548, 2480],
       [2235,  334, 1721],
       [2207, 1801, 2000],
       [ 684, 1895, 2582],
       [2149, 1321, 1354],
       [1056,  580,   45],
       [2424, 1346,  145],
       [2159,  145, 2582],
       [ 752,  498, 2246],
       [2014, 1687,  543],
       [1687, 1590,  543],
       [1193, 1033,    4],
       [1333, 2234, 2082],
       [2230, 1291, 1033],
       [ 238,  870,    4],
       [ 543, 1687, 2485],
       [1372,  493,   91],
       [1396, 1970, 2341],
       [1301,  318, 2236],
       [1506,  791, 2577],
       [1687, 2460, 2485],
       [ 615,  230, 1800],
       [2537,  474, 2696],
       [1490, 1312,  543],
       [ 357,  237, 1454],
       [ 187, 2244, 2597],
       [ 543, 1687,    4],
       [ 741, 1999, 1201],
       [1685, 1747, 1033],
       [2588,  543, 1747],
       [1305, 1033,    4],
       [1938,  925,  924],
       [1305,  543, 2485],
       [1170, 1687,    4],
       [ 653,  728, 1782],
       [1266,   90,  817],
       [2626,  703, 2475],
       [1416, 1279, 2305],
       [2711,  703, 2475],
       [ 113, 1177, 1769],
       [1225, 2166,  820],
       [1272,  881,  297],
       [2638,  880,  674],
       [2129,  960, 1500],
       [ 351,  903,  774],
       [1731, 2240, 2446],
       [ 642, 2137, 2475],
       [2076, 1371,  369],
       [  72,  703, 2475],
       [1915, 2070, 2475],
       [1546,  334, 2480],
       [ 707, 1913, 1989],
       [2215,  872, 2669],
       [1999,  488, 1182],
       [1291,  508, 2208],
       [2367, 1361, 2286],
       [ 918, 1670,  179],
       [2215, 1713, 1670],
       [1649, 2831, 1142],
       [1768, 1268, 1361],
       [ 434,  137, 2535],
       [2081,  181, 2140],
       [1867,  664, 1219],
       [1344, 1056, 1546],
       [1556,  197, 1123],
       [1077, 1546,  709],
       [2086, 2390,   89],
       [1175,  160, 1176],
       [1057, 1315, 1088],
       [1110, 1260,  454],
       [1546, 2022, 1457],
       [1127, 1045, 2085],
       [ 630, 2304,  533],
       [1019,  280, 1453],
       [1291, 2304, 2327],
       [2469,  716,  233],
       [ 549,  884,  844],
       [2701,  493, 2449],
       [ 237,  295,  296],
       [ 754,  327, 1729],
       [1683,  702,  466],
       [1061, 1316,  714],
       [ 334, 1987, 1801],
       [2622, 1291,  295],
       [ 237, 1407,   51],
       [ 654, 1636, 2453],
       [1155, 2569, 1737],
       [1037, 1494, 2488],
       [ 419,  887, 1271],
       [1277,  676,  661],
       [ 555,  237, 1232],
       [ 445,  177, 1560],
       [ 237,   51, 1687],
       [1600, 1621, 2102],
       [ 162, 1566, 1864],
       [  46,  803,  646]], dtype=int64)
</pre>

```python
top_3_word = np.argsort(tf_idf_vect.toarray())[:, -3:]
df['important_word_indexes'] = pd.Series(top_3_word.tolist())
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page_no</th>
      <th>scene_title</th>
      <th>text</th>
      <th>processed_text</th>
      <th>important_word_indexes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1         EXT. MERCEDES WINDSHIELD -- DUSK</td>
      <td>1                It's raining...             ...</td>
      <td>its raining light strobes across the wet glas...</td>
      <td>[1984, 2387, 1978]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A1        INT. MERCEDES -- NIGHT</td>
      <td>A1                On his knee -- a syringe an...</td>
      <td>a on his knee a syringe and a gun the eyes of...</td>
      <td>[1297, 1971, 1097]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2         INT. COTTAGE BEDROOM -- NIGHT</td>
      <td>2                BOURNE'S EYES OPEN! -- panic...</td>
      <td>bournes eyes open panicked gasping trying to ...</td>
      <td>[1693, 2221, 968]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A2        INT. COTTAGE LIVING AREA/BATHROOM ...</td>
      <td>A2                BOURNE moving for the medic...</td>
      <td>a bourne moving for the medicine cabinet digs...</td>
      <td>[690, 299, 1482]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3         INT./EXT. COTTAGE LIVING ROOM/VERA...</td>
      <td>3                One minute later.  BOURNE mo...</td>
      <td>one minute later bourne moves out onto the ve...</td>
      <td>[2823, 1951, 1454]</td>
    </tr>
  </tbody>
</table>
</div>



```python
def convert_to_word(x):
    word_list = []
    for word in x:
        word_list.append(invert_index_vectorizer[word])
    return word_list
```


```python
df['important_words'] = df['important_word_indexes'].apply(lambda x: convert_to_word(x))
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page_no</th>
      <th>scene_title</th>
      <th>text</th>
      <th>processed_text</th>
      <th>important_word_indexes</th>
      <th>important_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1         EXT. MERCEDES WINDSHIELD -- DUSK</td>
      <td>1                It's raining...             ...</td>
      <td>its raining light strobes across the wet glas...</td>
      <td>[1984, 2387, 1978]</td>
      <td>[riding, strobes, rhythmic]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A1        INT. MERCEDES -- NIGHT</td>
      <td>A1                On his knee -- a syringe an...</td>
      <td>a on his knee a syringe and a gun the eyes of...</td>
      <td>[1297, 1971, 1097]</td>
      <td>[knee, returns, head]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2         INT. COTTAGE BEDROOM -- NIGHT</td>
      <td>2                BOURNE'S EYES OPEN! -- panic...</td>
      <td>bournes eyes open panicked gasping trying to ...</td>
      <td>[1693, 2221, 968]</td>
      <td>[panicked, sleeps, gasping]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A2        INT. COTTAGE LIVING AREA/BATHROOM ...</td>
      <td>A2                BOURNE moving for the medic...</td>
      <td>a bourne moving for the medicine cabinet digs...</td>
      <td>[690, 299, 1482]</td>
      <td>[downs, cabinet, medicine]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3         INT./EXT. COTTAGE LIVING ROOM/VERA...</td>
      <td>3                One minute later.  BOURNE mo...</td>
      <td>one minute later bourne moves out onto the ve...</td>
      <td>[2823, 1951, 1454]</td>
      <td>[write, remember, marie]</td>
    </tr>
  </tbody>
</table>
</div>



```python
```
