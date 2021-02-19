/**
 * Created by pranjal on 20-09-2020.
 */
var app = require('express')();
const session = require('express-session');

var GOOGLE_CLIENT_ID = '95188641970-b5dpuie1d0kevhgecfogpjfptfank8k9.apps.googleusercontent.com' ;
var GOOGLE_CLIENT_SECRET = 'aaPScCfl7yVlUS1aDRp6RIR0' ;
const GoogleStrategy = require('passport-google-oauth20').Strategy ;

const GithubStrategy = require('passport-github2');
var GITHUB_CLIENT_ID = '71c3fb7a0b6e5f92da0e' ;
var GITHUB_CLIENT_SECRET = '2206888708d5150e633fa3c3ba4ea4adff3dfdbc' ;

const TwitterStrategy = require('passport-twitter').Strategy;
var TWITTER_CONSUMER_KEY = 'GfGVfvQoNq9ukDp8vcm9Q1eS9' ;
var TWITTER_CONSUMER_SECRET = 'nnXtEVDMe9DwciJRu7EsFkhZD96yDRB119KAD8bD8zN9ht6azd' ;

/*
 BEARER TOKEN =AAAAAAAAAAAAAAAAAAAAAHwMIAEAAAAAHLrc7Xq03KnAQ%2BV6MyGsB2DoNyI%3DYBl2uxpwPVSIolIa0c9KgL4p7StLi1n61IXtmLb9kI6yJWG5sC

const LinkedInStrategy = require('passport-linkedin-oauth2').Strategy;
var LINKEDIN_KEY = ''
var LINKEDIN_SECRET = ''
 */



const mongoose = require('mongoose');



require('./Models/DLparams.model');
var User = mongoose.model('User')

PassportStrategy = function(passport)
{
    passport.use(new GoogleStrategy({
        clientID:     GOOGLE_CLIENT_ID,
        clientSecret: GOOGLE_CLIENT_SECRET,
        callbackURL: '/auth/google/callback'
        //passReqToCallback   : true
    },
        async (accessToken, refreshToken, profile, done) => {
        console.log(profile);
        const newUser = {
            Id: profile.id,
            UserName: profile.displayName,
            Name: profile.name.givenName + profile.name.familyName,
            image: profile.photos[0].value
        }

        try
        {
            let user = await User.findOne({ Id: profile.id });
            if (user)
            {
                done(null, user)
            }
            else
            {
                user = await User.create(newUser)
                done(null, user)
            }
}       catch (err)
        {
            console.error(err)
        }

    }))

    passport.use(new GithubStrategy({
        clientID : GITHUB_CLIENT_ID,
        clientSecret : GITHUB_CLIENT_SECRET,
        callback : '/auth/github/callback'
    },
    async (accessToken, refreshToken, profile, done ) => {
        console.log('Github profile :\t' + JSON.stringify(profile)) ;
        const newUser = {
            Id: profile.id,
            UserName: profile.username,
            Name: profile.displayName,
            image: profile.photos[0].value
        }

        try
        {
            let user = await User.findOne({ Id: profile.id });
            if (user)
            {
                done(null, user)
            }
            else
            {
                user = await User.create(newUser)
                done(null, user)
            }
}       catch (err)
        {
            console.error(err)
        }

        return done(null, profile);
    }))

    passport.use(new TwitterStrategy({
        consumerKey: TWITTER_CONSUMER_KEY,
        consumerSecret: TWITTER_CONSUMER_SECRET,
        callbackURL: "/auth/twitter/callback"
    },
    async (token, tokenSecret, profile, done) => {

         const newUser = {
            Id: String(profile._json.id),
            UserName: profile._json.screen_name,
            Name: profile._json.name,
            image: profile.photos[0].value
            }
         console.log(newUser)
        try
        {
            console.log('checking for new user\n');
            let user =  await User.findOne({ Id: profile._json.id });
            console.log(user) ;
            if (user)
            {
                done(null, user)
                console.log('found\n');
            }
            else
            {
                console.log('Creating New User\n');
                user = await User.create(newUser);
                console.log('Created....\n');
                done(null, user)
            }
        }       catch (err)
        {
            console.log('found\n');
            console.error(err)
        }

        return 0;
      //  User.findOrCreate({ twitterId: profile.id }, function (err, user) {
    }
));



    passport.serializeUser(function(user, done) {
       return done(null, user);
    });

    passport.deserializeUser(function(obj, done) {
       return done(null, obj);
    });
/*
 passport.serializeUser((user, done) => {
 done(null, user.id)
 })

 passport.deserializeUser((id, done) => {
 User.findById(id, (err, user) => done(err, user))
 })


 */
};


module.exports = PassportStrategy ;

